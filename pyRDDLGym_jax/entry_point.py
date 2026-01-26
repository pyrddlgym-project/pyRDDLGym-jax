import argparse


EPILOG = 'For complete documentation, see https://pyrddlgym.readthedocs.io/en/latest/jax.html.'

def main():
    parser = argparse.ArgumentParser(prog='jaxplan',
                                     description="command line parser for the jaxplan planner",
                                     epilog=EPILOG)
    subparsers = parser.add_subparsers(dest="jaxplan", required=True)

    # planning
    parser_plan = subparsers.add_parser("plan", 
                                        help="execute jaxplan on a specified RDDL problem",
                                        epilog=EPILOG)
    parser_plan.add_argument('domain', type=str,  
                             help='name of domain in rddlrepository or a valid file path')
    parser_plan.add_argument('instance', type=str, 
                             help='name of instance in rddlrepository or a valid file path')
    parser_plan.add_argument('method', type=str, 
                             help='training method to apply: [slp, drp] are offline methods, and [replan] are online')
    parser_plan.add_argument('-e', '--episodes', type=int, required=False, default=1, 
                             help='number of training or evaluation episodes')

    # tuning
    parser_tune = subparsers.add_parser("tune", 
                                        help="tune jaxplan on a specified RDDL problem",
                                        epilog=EPILOG)
    parser_tune.add_argument('domain', type=str,  
                             help='name of domain in rddlrepository or a valid file path')
    parser_tune.add_argument('instance', type=str, 
                             help='name of instance in rddlrepository or a valid file path')
    parser_tune.add_argument('method', type=str, 
                             help='training method to apply: [slp, drp] are offline methods, and [replan] are online')
    parser_tune.add_argument('-t', '--trials', type=int, required=False, default=5, 
                             help='number of evaluation rollouts per hyper-parameter choice')
    parser_tune.add_argument('-i', '--iters', type=int, required=False, default=20, 
                             help='number of iterations of bayesian optimization')
    parser_tune.add_argument('-w', '--workers', type=int, required=False, default=4, 
                             help='number of parallel hyper-parameters to evaluate per iteration')
    parser_tune.add_argument('-d', '--dashboard', type=bool, required=False, default=False, 
                             help='show the dashboard')
    parser_tune.add_argument('-f', '--filepath', type=str, required=False, default='', 
                             help='where to save the config file of the best hyper-parameters')

    # dispatch
    args = parser.parse_args()
    if args.jaxplan == "plan":
        from pyRDDLGym_jax.examples import run_plan
        run_plan.main(args.domain, args.instance, args.method, args.episodes)
    elif args.jaxplan == "tune":
        from pyRDDLGym_jax.examples import run_tune
        run_tune.main(args.domain, args.instance, args.method, 
                      args.trials, args.iters, args.workers, args.dashboard, 
                      args.filepath)
    else:
        parser.print_help()

if __name__ == "__main__": 
    main()
