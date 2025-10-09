import logging
from aide import Experiment
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG,  # This sets the global level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("aide").setLevel(logging.DEBUG)

load_dotenv(override=True)
exp = Experiment(data_dir="input", goal="Predict the sales price for each house", eval="Use the RMSE metric between the logarithm of the predicted and observed values.")

exp.run(3)