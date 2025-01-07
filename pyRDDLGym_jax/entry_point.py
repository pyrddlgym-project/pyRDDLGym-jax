import argparse

from pyRDDLGym_jax.examples import run_plan, run_tune

def main():
    parser = argparse.ArgumentParser(description="Command line parser for the JaxPlan planner.")
    subparsers = parser.add_subparsers(dest="jaxplan", required=True)

    # planning
    parser_plan = subparsers.add_parser("plan", help="Executes JaxPlan on a specified RDDL problem and method (slp, drp, or replan).")
    parser_plan.add_argument('args', nargs=argparse.REMAINDER)

    # tuning
    parser_tune = subparsers.add_parser("tune", help="Tunes JaxPlan on a specified RDDL problem and method (slp, drp, or replan).")
    parser_tune.add_argument('args', nargs=argparse.REMAINDER)

    # dispatch
    args = parser.parse_args()
    if args.jaxplan == "plan":
        run_plan.run_from_args(args.args)
    elif args.jaxplan == "tune":
        run_tune.run_from_args(args.args)
    else:
        parser.print_help()

if __name__ == "__main__": 
    main()
