from cement.core.foundation import CementApp
from dqn.dqnLunar import DQNLunar

with CementApp("dqnTest") as app:

    app.args.add_argument('-e', '--env', action="store", dest="env", help="specify an environment to run the DQN on")

    app.run()
    print("DQN Testing Framework")

    if app.pargs.env:
        print("Received environment: %s" %app.pargs.env)
        environment = app.pargs.env.lower()
        if environment in ("lunar", "lunarlander"):
            scenario = DQNLunar()
            scenario.run()
        else:
            print("Scenario not found.")