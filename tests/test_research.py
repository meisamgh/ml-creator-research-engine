import unittest
import json
import os
import tempfile
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from ml_creator.__main__ import run_config, write_json_atomic
from ml_creator.core import Problem
from ml_creator.research import (FeatureResearchRun, StructureResearchRun, _json_request,
                                 _review_structure_candidate, research_features, research_structures, reviewer_config,
                                 search, source_type_for_url)


URL = "https://example.org/study"
SOURCE = [{"url": URL, "title": "Study", "snippet": "Method summary", "evidence_level": "SNIPPET_ONLY"}]
CONFIG = {"problem": {"task": "probability_of_default", "entity": "customer", "horizon_days": 365},
          "intent": {"mode": "STRUCTURES", "data_state": "NO_DATA"},
          "research": {"requested_features": 1, "requested_model_structures": 1}}


class ResearchTests(unittest.TestCase):
    def test_structure_and_feature_modes(self):
        model = {"name": "regularized_baseline", "problem_type": "binary classification",
                 "architecture": "regularized logistic regression", "components": ["imputation", "logistic regression"],
                 "rationale": "interpretable baseline", "assumptions": ["binary label"],
                 "required_data_properties": ["historical labeled records"],
                 "training_protocol": "fit preprocessing within training folds",
                 "validation_protocol": "time-blocked OOF and sealed holdout",
                 "strengths": ["simple"], "risks": ["missed nonlinearities"], "sources": [URL]}
        feature = {"concept_id": "payment_irregularity", "name": "Payment irregularity",
                   "concept_family": "payment behavior", "mechanism": "timing instability",
                   "required_information": ["payment timestamp"], "typical_formula": "past interval variance",
                   "source_ids": [URL]}
        with patch("ml_creator.research.search", return_value=SOURCE), patch(
                "ml_creator.research._ask_for_json", side_effect=lambda p, s, k, n, *rest: [feature if k == "feature" else model]), patch(
                "ml_creator.research._plan_structure_round", return_value=("query one", "query two")), patch(
                "ml_creator.research._critique_structure_round", return_value={"gaps": (), "reason": "covered"}), patch(
                "ml_creator.research.reviewer_config", return_value=("https://review.example", "different-model", "key", "anthropic")), patch(
                "ml_creator.research._review_structure_candidate", return_value={"is_model_structure": True,
                    "is_new": True, "source_support": "DIRECT", "canonical_structure": "logistic regression", "reason": "new"}), patch(
                "ml_creator.research._plan_feature_round", return_value=("feature query",)), patch(
                "ml_creator.research._critique_feature_round", return_value={"gaps": (), "reason": "covered"}):
            structures = run_config(CONFIG)
            self.assertEqual(structures["structure_research"][0]["status"], "RESEARCH_PROPOSAL")
            self.assertNotIn("feature_research", structures)
            features = run_config(CONFIG, mode_override="FEATURES")
            self.assertEqual(features["feature_research"][0]["source_ids"], (URL,))
            self.assertNotIn("structure_research", features)
            both = run_config(CONFIG, mode_override="BOTH")
            self.assertEqual(len(both["structure_research"]), 1)
            self.assertEqual(len(both["feature_research"]), 1)
            self.assertEqual(both["source_evidence_level"], "SNIPPET_ONLY")
            self.assertEqual(both["structure_agent"]["stop_reason"], "TARGET_REACHED")

    def test_unknown_sources_are_rejected(self):
        proposal = {"concept_id": "x", "name": "X", "concept_family": "x", "mechanism": "x",
                    "required_information": ["x"], "typical_formula": "x", "source_ids": ["https://invented.test"]}
        with patch("ml_creator.research.search", return_value=SOURCE), patch(
                "ml_creator.research._ask_for_json", return_value=[proposal]), patch(
                "ml_creator.research._plan_feature_round", return_value=("query",)), patch(
                "ml_creator.research._critique_feature_round", return_value={"gaps": (), "reason": "none"}):
            self.assertEqual(research_features(Problem("default", "customer", 365), max_rounds=2).proposals, ())

    def test_data_mode_fails_before_research(self):
        config = {**CONFIG, "intent": {"mode": "STRUCTURES", "data_state": "DATA_AVAILABLE"}}
        with self.assertRaisesRegex(ValueError, "dataset adapter"):
            run_config(config)
        with self.assertRaisesRegex(ValueError, "problem and intent objects"):
            run_config({"problem": []})

    def test_feature_count_override_enables_feature_only_run(self):
        config = {**CONFIG, "research": {"requested_features": 0, "requested_model_structures": 100}}
        with self.assertRaisesRegex(ValueError, "requested_features > 0"):
            run_config(config, mode_override="FEATURES")
        with patch("ml_creator.__main__.research_features",
                   return_value=FeatureResearchRun((), (), 1, "MAX_ROUNDS")) as seeker:
            result = run_config(config, mode_override="FEATURES", feature_count_override=1)
        self.assertEqual(result["feature_agent"]["requested"], 1)
        self.assertEqual(seeker.call_args.kwargs["count"], 1)
        self.assertEqual(seeker.call_args.kwargs["source_types"], ("KAGGLE", "PAPERS", "REPOS"))

    def test_sources_override_routes_selected_categories(self):
        with patch("ml_creator.__main__.research_structures",
                   return_value=StructureResearchRun((), (), 1, "MAX_ROUNDS")) as seeker:
            result = run_config(CONFIG, sources_override="papers,repos")
        self.assertEqual(result["research_sources"], ("PAPERS", "REPOS"))
        self.assertEqual(seeker.call_args.kwargs["source_types"], ("PAPERS", "REPOS"))

    def test_problem_override_clears_old_horizon_for_every_mode(self):
        with patch("ml_creator.__main__.research_features", return_value=FeatureResearchRun((), (), 1, "MAX_ROUNDS")) as feature_seeker, patch(
                "ml_creator.__main__.research_structures", return_value=StructureResearchRun((), (), 1, "MAX_ROUNDS")) as structure_seeker:
            for mode in ("FEATURES", "STRUCTURES", "BOTH"):
                result = run_config(CONFIG, mode_override=mode, problem_override="customer_churn")
                self.assertEqual(result["problem"], {"task": "customer_churn", "entity": "customer", "horizon_days": None})
            self.assertIsNone(feature_seeker.call_args.args[0].horizon_days)
            self.assertIsNone(structure_seeker.call_args.args[0].horizon_days)
            result = run_config(CONFIG, mode_override="BOTH", problem_override="customer_churn", horizon_override=90)
            self.assertEqual(result["problem"]["horizon_days"], 90)

    def test_optional_horizon_validation(self):
        self.assertIsNone(Problem("churn", "customer").horizon_days)
        with self.assertRaisesRegex(ValueError, "positive"):
            Problem("churn", "customer", 0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            Problem("churn", "customer", "wrong")
        no_horizon_config = {**CONFIG, "problem": {**CONFIG["problem"], "horizon_days": "no-horizon"}}
        with patch("ml_creator.__main__.research_structures",
                   return_value=StructureResearchRun((), (), 1, "MAX_ROUNDS")):
            self.assertIsNone(run_config(no_horizon_config)["problem"]["horizon_days"])
        with patch("ml_creator.__main__.research_features",
                   return_value=FeatureResearchRun((), (), 1, "MAX_ROUNDS")), patch(
                   "ml_creator.__main__.research_structures",
                   return_value=StructureResearchRun((), (), 1, "MAX_ROUNDS")):
            for mode in ("FEATURES", "STRUCTURES", "BOTH"):
                result = run_config(CONFIG, mode_override=mode, no_horizon=True)
                self.assertIsNone(result["problem"]["horizon_days"])
        with self.assertRaisesRegex(ValueError, "not both"):
            run_config(CONFIG, no_horizon=True, horizon_override=30)

    def test_transient_provider_error_retries_without_key_in_argv(self):
        calls = []
        def fake_curl(argv, **kwargs):
            calls.append((argv, kwargs["input"]))
            if len(calls) == 1:
                return CompletedProcess(argv, 0, "520 text/plain", "")
            Path(argv[argv.index("--output") + 1]).write_text(json.dumps({"type": "message"}))
            return CompletedProcess(argv, 0, "200 application/json", "")
        with patch("ml_creator.research.subprocess.run", side_effect=fake_curl), patch(
                "ml_creator.research.time.sleep"):
            result = _json_request("https://example.org/v1/messages", payload={"x": 1},
                                   headers={"x-api-key": "secret-test-key"})
        self.assertEqual(result["type"], "message")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("secret-test-key", " ".join(calls[0][0]))

    def test_large_structure_request_is_bounded_and_auditable(self):
        model = {"name": "baseline", "problem_type": "binary classification", "architecture": "logistic",
                 "components": ["logistic"], "rationale": "baseline", "assumptions": ["binary target"],
                 "required_data_properties": ["labels"], "training_protocol": "fit on past",
                 "validation_protocol": "time blocks", "strengths": ["simple"], "risks": ["bias"],
                 "sources": [URL]}
        progress = []
        with patch("ml_creator.research.search", return_value=SOURCE), patch(
                "ml_creator.research._ask_for_json", return_value=[model]), patch(
                "ml_creator.research._plan_structure_round", side_effect=[("first",), ("second",)]) as planner, patch(
                "ml_creator.research._critique_structure_round", return_value={"gaps": ("nonlinear models",), "reason": "gap"}), patch(
                "ml_creator.research.reviewer_config", return_value=("https://review.example", "different-model", "key", "anthropic")), patch(
                "ml_creator.research._review_structure_candidate", return_value={"is_model_structure": True,
                    "is_new": True, "source_support": "DIRECT", "canonical_structure": "logistic regression", "reason": "new"}):
            run = research_structures(Problem("default", "customer", 365), count=100, max_rounds=2,
                                      on_round=progress.append)
        self.assertEqual(len(run.proposals), 1)
        self.assertEqual(run.stop_reason, "MAX_ROUNDS")
        self.assertEqual(len(run.rounds), 2)
        self.assertEqual([p.stop_reason for p in progress], ["IN_PROGRESS", "IN_PROGRESS"])
        self.assertEqual(planner.call_args.kwargs["previous_gaps"], ("nonlinear models",))

    def test_source_domain_filter(self):
        self.assertEqual(source_type_for_url("https://www.kaggle.com/code/example"), "KAGGLE")
        self.assertEqual(source_type_for_url("https://arxiv.org/abs/1234"), "PAPERS")
        self.assertEqual(source_type_for_url("https://github.com/org/repo"), "REPOS")
        self.assertIsNone(source_type_for_url("https://github.com.evil.test/repo"))
        items = {"results": [
            {"url": "https://arxiv.org/abs/1234", "title": "Paper", "content": "architecture"},
            {"url": "https://github.com/org/repo", "title": "Repo", "content": "code"}]}
        with patch("ml_creator.research._json_request", return_value=items):
            found = search(Problem("default", "customer"), "structure", queries=("hybrid model",),
                           source_types=("PAPERS",))
        self.assertEqual([s["source_type"] for s in found], ["PAPERS"])

    def test_reviewer_defaults_to_separate_justwoker_call(self):
        env = {"JUSTWOKER_MODEL": "generator", "JUSTWOKER_BASE_URL": "https://justwoker.example",
               "JUSTWOKER_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True), patch("ml_creator.research._load_local_env"):
            self.assertEqual(reviewer_config(), ("https://justwoker.example", "generator", "test-key", "anthropic"))
            os.environ["REVIEW_MODEL"] = "reviewer"
            self.assertEqual(reviewer_config()[1], "reviewer")

    def test_separate_reviewer_controls_structure_admission(self):
        def proposal(name, architecture):
            return {"name": name, "problem_type": "binary classification",
                    "architecture": architecture, "components": ["XGBoost", "LSTM"],
                    "rationale": "tabular and sequence information", "assumptions": [],
                    "required_data_properties": ["transactions"], "strengths": [], "risks": [],
                    "sources": [URL], "validation_protocol": "brief time split"}
        proposed = [proposal("XGBoost plus LSTM", "XGBoost and LSTM branches combined"),
                    proposal("LSTM XGB hybrid", "LSTM plus boosted trees"),
                    proposal("Time-blocked validation", "time-blocked cross-validation")]
        first = {"is_model_structure": True, "is_new": True,
                 "source_support": "ADAPTED", "canonical_structure": "XGBoost plus LSTM hybrid", "reason": "new architecture"}
        second = {"is_model_structure": True, "is_new": False,
                  "source_support": "DIRECT", "canonical_structure": "XGBoost plus LSTM hybrid", "reason": "same component flow"}
        third = {"is_model_structure": False, "is_new": True,
                 "source_support": "NONE", "canonical_structure": "time-blocked validation", "reason": "validation procedure only"}
        with patch("ml_creator.research.reviewer_config", return_value=("https://review.example", "reviewer", "key", "anthropic")), patch(
                "ml_creator.research.search", return_value=SOURCE), patch(
                "ml_creator.research._plan_structure_round", return_value=("hybrid",)), patch(
                "ml_creator.research._ask_for_json", return_value=proposed), patch(
                "ml_creator.research._review_structure_candidate", side_effect=[first, second, third]) as reviewer, patch(
                "ml_creator.research._critique_structure_round", return_value={"gaps": (), "reason": "checked"}):
            run = research_structures(Problem("default", "customer"), count=3, max_rounds=1)
        self.assertEqual(reviewer.call_count, 3)
        self.assertEqual(len(run.proposals), 1)
        self.assertEqual([x["decision"] for x in run.rounds[0]["admission_reviews"]], ["ACCEPT", "REJECT", "REJECT"])
        self.assertEqual(run.proposals[0].validation_protocol, "brief time split")
        self.assertEqual(run.proposals[0].canonical_structure, "XGBoost plus LSTM hybrid")

    def test_malformed_model_fields_never_reach_reviewer(self):
        malformed = {"name": "Hybrid", "problem_type": "binary", "architecture": "XGBoost plus LSTM",
                     "components": "XGBoost,LSTM", "rationale": "two signals", "sources": [URL]}
        with patch("ml_creator.research.reviewer_config", return_value=("https://review.example", "model", "key", "anthropic")), patch(
                "ml_creator.research.search", return_value=SOURCE), patch(
                "ml_creator.research._plan_structure_round", return_value=("hybrid",)), patch(
                "ml_creator.research._ask_for_json", return_value=[malformed]), patch(
                "ml_creator.research._review_structure_candidate") as reviewer, patch(
                "ml_creator.research._critique_structure_round", return_value={"gaps": (), "reason": "invalid"}):
            run = research_structures(Problem("default", "customer"), count=1, max_rounds=1)
        self.assertEqual(run.proposals, ())
        self.assertEqual(run.rounds[0]["rejected_count"], 1)
        reviewer.assert_not_called()

    def test_reviewer_requires_source_support_verdict(self):
        from ml_creator.core import ModelStructure
        candidate = ModelStructure("hybrid", "binary", "XGBoost plus LSTM", ("XGBoost", "LSTM"),
                                   "two data types", (), (), "", "", (), (), (URL,))
        answer = {"is_model_structure": True, "is_new": True,
                  "canonical_structure": "XGBoost plus LSTM", "reason": "new"}
        with patch("ml_creator.research._review_model_json", return_value=answer):
            with self.assertRaisesRegex(ValueError, "source support"):
                _review_structure_candidate(candidate, (), tuple(SOURCE))

    def test_artifact_write_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("old")
            write_json_atomic(path, {"status": "NO_DATA", "count": 2})
            self.assertEqual(json.loads(path.read_text()), {"status": "NO_DATA", "count": 2})
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["result.json"])


if __name__ == "__main__":
    unittest.main()
