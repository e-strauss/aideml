import logging
from aide import Experiment
from dotenv import load_dotenv

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # This sets the global level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("aide").setLevel(logging.DEBUG)

    load_dotenv(override=True)
    exp = Experiment(data_dir="input", goal="the target column is the 'Price', start very simple and try different models, then continue to improve the feature engineering", eval="use cv with 5 folds")
    # with open("task_description.txt","r") as f:
    #    goal = f.read()
    # exp = Experiment(data_dir="input", goal=goal)

    exp.run(30)
