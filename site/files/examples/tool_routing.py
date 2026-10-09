"""Choose a support tool, deferring requests below a probability threshold."""
import json
import math

from jevany import Choice
from jevany.api import validate_distribution
from jevany.client import DecisionClient
from ._common import client, parser

TOOLS = {
    "order_status": "Look up shipment and delivery status for an existing order.",
    "return_policy": "Look up return eligibility, return deadlines and refund rules.",
    "product_search": "Find products matching the customer's requirements.",
    "reply": "Respond directly when no lookup is needed, or ask for missing information.",
}
MESSAGE = "Where is order A-104? It was due yesterday and has not arrived."


def run(decide: DecisionClient, message: str = MESSAGE, *, min_probability: float = 0.8) -> dict:
    """Return a tool name or a deferred decision; execution belongs to the caller."""
    if not math.isfinite(min_probability) or not 0 <= min_probability <= 1:
        raise ValueError("min_probability must be finite and in [0, 1]")
    response = decide.system_one(
        state={"message": message},
        questions={"tool": Choice(
            instructions="Choose the next tool for this request. Choose reply if a lookup is unnecessary "
                         "or the information needed for a lookup is missing.",
            criteria=TOOLS,
        )},
    )
    answer = response["answers"]["tool"]
    keys = list(TOOLS)
    probabilities, _ = validate_distribution(answer["probabilities"], keys)
    probability = probabilities[keys.index(answer["choice"])]
    tool = answer["choice"] if probability >= min_probability else None
    return {
        "tool": tool, "deferred": tool is None, "probability": probability,
        "min_probability": min_probability, "decision": response,
    }


def main() -> None:
    arguments = parser(__doc__)
    arguments.add_argument("message", nargs="?", default=MESSAGE)
    arguments.add_argument("--min-probability", type=float, default=0.8)
    args = arguments.parse_args()
    if not math.isfinite(args.min_probability) or not 0 <= args.min_probability <= 1:
        arguments.error("--min-probability must be finite and in [0, 1]")
    print(json.dumps(run(client(args), args.message, min_probability=args.min_probability), indent=2))


if __name__ == "__main__":
    main()
