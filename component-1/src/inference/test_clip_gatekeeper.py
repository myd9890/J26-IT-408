import argparse
import json

from src.inference.clip_gatekeeper import BananaLeafGatekeeper


def main():

    parser = argparse.ArgumentParser(
        description="Test the zero-shot banana-leaf gatekeeper."
    )

    parser.add_argument(
        "image_path",
        help="Path to the image to test.",
    )

    args = parser.parse_args()

    gatekeeper = BananaLeafGatekeeper()

    result = gatekeeper.predict(args.image_path)

    print("\n========== CLIP GATEKEEPER RESULT ==========")

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()