import subprocess
import sys
import os

DOCKER_IMAGE = "python_sandbox:latest"
CONTAINER_NAME = "sandbox"
DOCKERFILE_DIR = os.path.join(os.path.dirname(__file__), "resources/docker")


def image_exists(image_name):
    result = subprocess.run([
        "docker", "images", "-q", image_name
    ], capture_output=True, text=True)
    return result.stdout.strip() != ""


def container_running(container_name):
    result = subprocess.run([
        "docker", "ps", "-q", "-f", f"name=^{container_name}$"
    ], capture_output=True, text=True)
    return result.stdout.strip() != ""


def container_exists(container_name):
    result = subprocess.run([
        "docker", "ps", "-aq", "-f", f"name=^{container_name}$"
    ], capture_output=True, text=True)
    return result.stdout.strip() != ""


def build_image():
    print(f"Building Docker image '{DOCKER_IMAGE}'...")
    result = subprocess.run([
        "docker", "build", "-t", DOCKER_IMAGE, DOCKERFILE_DIR
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error building Docker image:")
        print(result.stderr)
        sys.exit(1)
    print("Docker image built successfully.")


def run_container():
    print(f"Running container '{CONTAINER_NAME}'...")
    # Remove any stopped container with the same name
    if container_exists(CONTAINER_NAME) and not container_running(CONTAINER_NAME):
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
    result = subprocess.run([
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--network", "none",
        "--cap-drop", "all",
        "--pids-limit", "64",
        "--tmpfs", "/tmp:rw,size=64M",
        DOCKER_IMAGE, "sleep", "infinity"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running Docker container:")
        print(result.stderr)
        sys.exit(1)
    print(f"Container started with ID: {result.stdout.strip()}")


def main():
    if not image_exists(DOCKER_IMAGE):
        build_image()
    else:
        print(f"Docker image '{DOCKER_IMAGE}' already exists.")

    if container_running(CONTAINER_NAME):
        print(f"Container '{CONTAINER_NAME}' is already running.")
        result = subprocess.run([
            "docker", "ps", "-q", "-f", f"name=^{CONTAINER_NAME}$"
        ], capture_output=True, text=True)
        print(f"Container ID: {result.stdout.strip()}")
    else:
        run_container()


if __name__ == "__main__":
    main() 