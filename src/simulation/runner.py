import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.analysis.market_mechanism import runner_4_1
from src.analysis.wealth_effect import runner_4_2
from src.analysis.market_efficiency import runner_4_3
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # runner_4_1()
    # runner_4_2()
    runner_4_3()
