#!/usr/bin/env python3
"""
Deploy Gradio app to HuggingFace Spaces.
"""

import argparse
import shutil
from pathlib import Path
from huggingface_hub import HfApi


def create_space_files(
    model_repo_id: str,
    space_dir: Path = Path("space_deployment"),
) -> Path:
    """
    Prepare files for Space deployment.

    Args:
        model_repo_id: Full model repo ID (e.g., "ccss17/modernbert-prompt-injection-detector")
        space_dir: Directory to prepare Space files

    Returns:
        Path to the prepared space directory
    """
    space_dir = Path(space_dir)
    space_dir.mkdir(exist_ok=True, parents=True)

    print(f"📁 Preparing Space files in: {space_dir}")

    # Copy app.py and update MODEL_NAME
    app_source = Path("app.py")
    if not app_source.exists():
        raise FileNotFoundError("app.py not found in current directory")

    app_content = app_source.read_text()
    # Update MODEL_NAME to deployed model
    # Handle both old and new format
    if 'MODEL_NAME = "answerdotai/ModernBERT-large"' in app_content:
        app_content = app_content.replace(
            'MODEL_NAME = "answerdotai/ModernBERT-large"  # Update this after deployment',
            f'MODEL_NAME = "{model_repo_id}"',
        )
    elif (
        'MODEL_NAME = "ccss17/modernbert-prompt-injection-detector"'
        in app_content
    ):
        app_content = app_content.replace(
            'MODEL_NAME = "ccss17/modernbert-prompt-injection-detector"',
            f'MODEL_NAME = "{model_repo_id}"',
        )
    else:
        # Add MODEL_NAME if not found
        print("⚠️  Warning: MODEL_NAME not found in app.py, using as-is")

    (space_dir / "app.py").write_text(app_content)
    print(f"✅ Updated app.py with model path: {model_repo_id}")

    # Copy requirements
    requirements_source = Path("requirements_spaces.txt")
    if requirements_source.exists():
        shutil.copy(requirements_source, space_dir / "requirements.txt")
        print("✅ Copied requirements.txt")
    else:
        print("⚠️  requirements_spaces.txt not found")

    # Copy/create README
    readme_source = Path("SPACE_README.md")
    if readme_source.exists():
        readme_content = readme_source.read_text()
        # Update model references
        readme_content = readme_content.replace(
            "ccss17/modernbert-prompt-injection-detector", model_repo_id
        )
        (space_dir / "README.md").write_text(readme_content)
        print("✅ Created README.md for Space")
    else:
        print("⚠️  SPACE_README.md not found")

    # Copy example prompts if exists
    examples_source = Path("example_prompts.txt")
    if examples_source.exists():
        shutil.copy(examples_source, space_dir / "example_prompts.txt")
        print("✅ Copied example_prompts.txt")

    return space_dir


def upload_space(
    space_id: str,
    space_dir: Path,
    token: str = None,
    private: bool = False,
):
    """
    Upload prepared Space to HuggingFace.

    Args:
        space_id: Space ID (format: username/space-name)
        space_dir: Directory containing Space files
        token: HuggingFace token (uses HF_TOKEN env var if not provided)
        private: Whether to make the Space private
    """
    print(f"\n🚀 Deploying Space: {space_id}")
    print(f"📂 From directory: {space_dir}")
    print(f"🔒 Private: {private}")

    # Initialize API
    api = HfApi(token=token)

    # Create Space
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            private=private,
            exist_ok=True,
        )
        print(f"✅ Space created/verified: {space_id}")
    except Exception as e:
        print(f"⚠️  Space creation warning: {e}")

    # Upload all files
    print("⬆️  Uploading Space files...")
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy Prompt Injection Detector demo",
    )

    print(f"\n🎉 Space successfully deployed!")
    print(f"🔗 View at: https://huggingface.co/spaces/{space_id}")
    print(f"\n⏳ Note: Space may take 1-2 minutes to build and start")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Gradio app to HuggingFace Spaces"
    )
    parser.add_argument(
        "--model-repo-id",
        type=str,
        required=True,
        help="Model repository ID (e.g., ccss17/modernbert-prompt-injection-detector)",
    )
    parser.add_argument(
        "--space-id",
        type=str,
        required=True,
        help="Space ID (format: username/space-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make Space private",
    )
    parser.add_argument(
        "--space-dir",
        type=str,
        default="space_deployment",
        help="Directory to prepare Space files",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare files without uploading",
    )

    args = parser.parse_args()

    # Prepare Space files
    space_dir = create_space_files(
        model_repo_id=args.model_repo_id,
        space_dir=Path(args.space_dir),
    )

    if args.prepare_only:
        print(f"\n✅ Files prepared in: {space_dir}")
        print(f"📝 Review the files and run without --prepare-only to upload")
        return

    # Upload to HuggingFace
    upload_space(
        space_id=args.space_id,
        space_dir=space_dir,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
