"""Compile typed questions with a local Chat Completions planner, then decide with JevAny."""
import json
import os

from ._common import client, parser
from jevany.harness import JevHarness
from jevany.openai_compat import ChatCompletionsGenerator


def main(argv: list[str] | None = None) -> None:
    arguments = parser(__doc__)
    arguments.add_argument("--planner-model", required=True, help="model name served by the planner")
    arguments.add_argument("--planner-url", default="http://127.0.0.1:8000/v1", help="planner API prefix")
    arguments.add_argument("--task", required=True)
    arguments.add_argument("--evidence", required=True, help="JSON object, array, or scalar")
    arguments.add_argument("--planner-sees-evidence", action="store_true",
                           help="send evidence values to the planner; the default sends only the schema")
    args = arguments.parse_args(argv)
    evidence = json.loads(args.evidence)
    planner = ChatCompletionsGenerator(
        args.planner_model, base_url=args.planner_url,
        api_key=os.environ.get("JEVANY_PLANNER_API_KEY"),
    )
    harness = JevHarness(planner, client(args), include_evidence_in_planner=args.planner_sees_evidence)
    print(json.dumps(harness.run(args.task, evidence), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
