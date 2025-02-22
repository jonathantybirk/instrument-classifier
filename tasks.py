import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "instrument_classifier"
PYTHON_VERSION = "3.13"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def profile_train(ctx: Context) -> None:
    """Profile training run and export Chrome trace."""
    ctx.run(f"python src/{PROJECT_NAME}/profiler.py", echo=True, pty=not WINDOWS)


@task
def train(
    ctx: Context,
    overrides: str = "",
    num_layers: int = 3,
    num_classes: int = 4,
    input_channels: int = 1,
    kernel_size: int = 3,
    dropout_rate: float = 0.5,
    batch_size: int = 32,
) -> None:
    """Train model with optional Hydra configuration overrides.


    Args:
        ctx: Invoke context
        overrides: Hydra configuration overrides (e.g. "model.lr=0.001 training.batch_size=32")
    """
    cmd = f"python src/{PROJECT_NAME}/train.py -m {overrides}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def cloud_train(ctx: Context) -> None:
    """Train model with optional Hydra configuration overrides.

    Args:
        ctx: Invoke context
        overrides: Hydra configuration overrides (e.g. "model.lr=0.001 training.batch_size=32")
    """
    cmd = (
        "gcloud ai custom-jobs create "
        "--region=europe-west4 "
        "--display-name=test-run "
        "--config=configs/config_gpu.yaml "
        "--command 'python src/{PROJECT_NAME}/train.py' "
        "--args=-m --args=--training.num_epochs=1,2"
    )
    import subprocess

    # ctx.run(cmd, echo=True, pty=not WINDOWS)
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def run_api(ctx: Context) -> None:
    """Run API."""
    ctx.run(
        "uvicorn instrument_classifier.api:app --reload --port 8000",
        echo=True,
        pty=not WINDOWS,
    )


@task
def healthcheck(ctx: Context) -> None:
    """Healthcheck API."""
    command = "curl.exe" if WINDOWS else "curl"
    ctx.run(f"{command} -s http://localhost:8000/health", echo=True, pty=not WINDOWS)


@task
def send_request(
    ctx: Context,
    path_to_audio: str = r"data\raw\train_submission\emotional-piano-001-d-90-66506.wav",
) -> None:
    """Send request to API."""
    command = "curl.exe" if WINDOWS else "curl"
    ctx.run(
        f'{command} -s -F "file=@{path_to_audio}" http://localhost:8000/predict',
        echo=True,
        pty=not WINDOWS,
    )


@task
def visualize(ctx: Context) -> None:
    """Visualize spectrograms."""
    ctx.run(f"python src/{PROJECT_NAME}/visualize.py", echo=True, pty=not WINDOWS)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(
        "mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS,
    )


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# Evaluate
@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    # Set environment variable to suppress the FutureWarning about torch.load
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    try:
        ctx.run(f"python src/{PROJECT_NAME}/evaluate.py", echo=True, pty=not WINDOWS)
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print("Please ensure:")
        print("- The model file exists at 'models/best_cnn_audio_classifier.pt'")
        print("- The evaluation dataset is properly set up")
        raise
