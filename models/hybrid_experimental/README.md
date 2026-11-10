# Experimental SmartHire Hybrid Model

This bundle is for offline experimental inference only. It is not integrated into FastAPI or the production prediction path.

The text, structured, and fusion estimators were reproduced using the Stage 2 grouped split and capped development sample (maximum five rows per resume-group/label combination). Hybrid fusion was trained on grouped out-of-fold text probabilities. The reproduced holdout confusion matrices were checked against the saved Stage 2 aggregate metrics before this bundle was saved.

Keyword-derived features are heuristic and do not establish candidate competence. The model must not be used as an employment decision without further independent validation and appropriate human review.

Load through `src.hybrid_predict.predict_hybrid`; bundle metadata includes feature ordering, labels, configuration, and versions.

Only load this Joblib bundle from a trusted local source. Joblib artifacts use
Python serialization and must not be accepted from untrusted uploads or
remote users.
