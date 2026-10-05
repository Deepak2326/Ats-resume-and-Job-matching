"""
sample_jobs.py
==============
Deterministic test fixtures: eight realistic job listings used by the
engine-level and AppTest suites so tests never depend on the live boards.

NOTE: this module is imported ONLY by tests — the application itself never
ships or displays sample listings (scraper.py is strictly live-only).
"""

MOCK_JOBS: list[dict] = [
    {
        "job_title": "Senior Machine Learning Engineer",
        "company": "NovaHealth AI",
        "location": "Bengaluru",
        "job_type": "fulltime",
        "job_url": "https://example.com/jobs/novahealth-ml",
        "site": "indeed",
        "date_posted": "2026-03-28",
        "description": (
            "NovaHealth AI is hiring a Senior Machine Learning Engineer to own "
            "clinical NLP pipelines end to end. You will design and deploy "
            "models for medical entity extraction, fine-tune "
            "transformers on de-identified EHR notes, and ship services with "
            "Docker on AWS. Requirements: 4+ years of production ML experience, "
            "strong Python, PyTorch, Docker, AWS, and hands-on knowledge of "
            "FHIR and HIPAA compliance. Nice to have: model monitoring, "
            "Deep Learning, MLOps, distributed training."
        ),
    },
    {
        "job_title": "Data Scientist",
        "company": "Quantica Labs",
        "location": "Austin, TX",
        "job_type": "fulltime",
        "job_url": "https://example.com/jobs/quantica-ds",
        "site": "linkedin",
        "date_posted": "2026-03-25",
        "description": (
            "Quantica Labs seeks a Data Scientist to turn marketplace data into "
            "pricing intelligence. You will build gradient-boosted models in "
            "scikit-learn, run controlled experiments, and partner with "
            "engineering on real-time scoring APIs. Requirements: 3+ years "
            "experience, Python, SQL, pandas, scikit-learn, statistics, and "
            "clear communication with product stakeholders. Nice to have: "
            "Airflow, Spark, Tableau, Kubernetes."
        ),
    },
    {
        "job_title": "ML Platform Engineer",
        "company": "Orbital Dynamics",
        "location": "Remote",
        "job_type": "contract",
        "job_url": "https://example.com/jobs/orbital-mlp",
        "site": "glassdoor",
        "date_posted": "2026-03-22",
        "description": (
            "Orbital Dynamics is contracting an ML Platform Engineer to "
            "harden our training infrastructure. You will own GPU scheduling on "
            "Kubernetes, build feature pipelines with Spark and Airflow, and "
            "standardise model packaging with Docker. Requirements: strong "
            "Python, Kubernetes, AWS, CI/CD, and infrastructure-as-code "
            "experience. Nice to have: Ray, MLflow, Terraform, observability "
            "tooling."
        ),
    },
    {
        "job_title": "NLP Research Engineer",
        "company": "Lexicon Systems",
        "location": "New York, NY",
        "job_type": "fulltime",
        "job_url": "https://example.com/jobs/lexicon-nlp",
        "site": "zip_recruiter",
        "date_posted": "2026-03-20",
        "description": (
            "Lexicon Systems is hiring an NLP Research Engineer to push our "
            "document-understanding stack. You will prototype retrieval-"
            "augmented generation, evaluate LLM outputs, and productionise the "
            "best ideas with Python and PyTorch. Requirements: NLP research "
            "experience, transformers, embeddings, evaluation design, Python. "
            "Nice to have: RAG, LangChain, vector databases, publications."
        ),
    },

    {
        "job_title": "Computer Vision Engineer",
        "company": "Skyline Robotics",
        "location": "Pittsburgh, PA",
        "job_type": "fulltime",
        "job_url": "https://example.com/jobs/skyline-cv",
        "site": "indeed",
        "date_posted": "2026-03-18",
        "description": (
            "Skyline Robotics builds autonomous inspection drones and needs a "
            "Computer Vision Engineer for detection and tracking models. You "
            "will train YOLO-style detectors, optimise for edge deployment "
            "with TensorRT, and own the data loop from field collection to "
            "retraining. Requirements: Python, PyTorch, OpenCV, CUDA, edge "
            "deployment. Nice to have: ROS, 3D vision, SLAM, C++."
        ),
    },
    {
        "job_title": "Data Engineer (Analytics)",
        "company": "BrightCart",
        "location": "Seattle, WA",
        "job_type": "parttime",
        "job_url": "https://example.com/jobs/brightcart-de",
        "site": "linkedin",
        "date_posted": "2026-03-15",
        "description": (
            "BrightCart is hiring a part-time Data Engineer to keep our "
            "analytics warehouse trustworthy. You will build dbt models, "
            "maintain Airflow DAGs, and improve data quality checks across "
            "Postgres and BigQuery. Requirements: SQL, Python, Airflow, dbt, "
            "data modeling. Nice to have: BigQuery, Kafka, Docker, Looker."
        ),
    },
    {
        "job_title": "MLOps Engineer",
        "company": "FinSight",
        "location": "Chicago, IL",
        "job_type": "fulltime",
        "job_url": "https://example.com/jobs/finsight-mlops",
        "site": "glassdoor",
        "date_posted": "2026-03-12",
        "description": (
            "FinSight seeks an MLOps Engineer to own the model lifecycle for "
            "our credit-risk platform. You will automate training and "
            "deployment with CI/CD, add drift monitoring, and keep latency "
            "budgets on our serving stack. Requirements: Python, Docker, "
            "Kubernetes, CI/CD, MLflow, monitoring. Nice to have: AWS "
            "SageMaker, Terraform, feature stores, BentoML."
        ),
    },
    {
        "job_title": "Machine Learning Intern",
        "company": "CampusAI",
        "location": "Remote",
        "job_type": "internship",
        "job_url": "https://example.com/jobs/campusai-intern",
        "site": "zip_recruiter",
        "date_posted": "2026-03-10",
        "description": (
            "CampusAI offers a 12-week Machine Learning Internship working on "
            "student-success prediction models. You will clean datasets, train "
            "baseline models with scikit-learn, and present findings to "
            "mentors weekly. Requirements: coursework in ML, Python, pandas, "
            "scikit-learn, curiosity. Nice to have: Jupyter, statistics, "
            "Git, SQL."
        ),
    },
]
