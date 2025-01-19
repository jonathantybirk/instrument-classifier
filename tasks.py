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
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


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
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )

@task
def run_api(ctx: Context) -> None:
    """Run API."""
    ctx.run("uvicorn instrument_classifier.api:app --reload --port 8000", echo=True, pty=not WINDOWS)

@task 
def healthcheck(ctx: Context) -> None:
    """Healthcheck API."""
    command = "curl.exe" if WINDOWS else "curl"
    ctx.run(f"{command} -s http://localhost:8000/health", echo=True, pty=not WINDOWS)

@task
def send_request(ctx: Context, path_to_audio: str = r"data\raw\train_submission\emotional-piano-001-d-90-66506.wav") -> None:
    """Send request to API."""
    command = "curl.exe" if WINDOWS else "curl"
    ctx.run(f'{command} -s -F "file=@{path_to_audio}" http://localhost:8000/predict', echo=True, pty=not WINDOWS)

# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
