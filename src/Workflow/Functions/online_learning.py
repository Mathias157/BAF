"""
Bi-Directional Coupling of Balmorel-Balmorel with Online-Learning

Balmorel investment optimisation is coupled to an operational run,
with a generative model (decoder-encoder neural network) generating 
timeslices for the investment run. 

The sum of loss of load expectation and energy not served
is used to train the generative model for each iteration.

Created on 13.08.2025
@author: Mathias Berg Rosendal, PhD Student and Atefeh Hemmati Golsefidi, PostDoc at DTU Management
"""
#%% ------------------------------- ###
###             0. CLI              ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import click
import os
from neural_network import pretrain, train
import subprocess as sp
import logging
import shlex
from datetime import datetime
from pathlib import Path

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def _setup_logger(scenario: str):
    """Create Logs/<scenario>_<timestamp>.log and return (logger, logfile_path)."""
    logs_dir = Path(__file__).resolve().parents[2] / "Logs"   # -> /.../src/Logs
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile = logs_dir / f"{scenario}_{ts}.log"

    logger = logging.getLogger(f"online_learning.{scenario}.{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, logfile


def _run(cmd: str, logger: logging.Logger, logfile: Path) -> int:
    """Run a shell command, append stdout/stderr to logfile, and log the command."""
    logger.info(f"$ {cmd}")
    qlog = shlex.quote(str(logfile))
    rc = sp.run(f"{cmd} >> {qlog} 2>&1", shell=True)
    if rc.returncode != 0:
        logger.warning(f"Command returned non-zero exit code {rc}: {cmd}")
    return rc



@click.command()
@click.pass_context
@click.option('--scenario-name', type=str, required=True,
              help='Name of the scenario to run')
@click.option('--scenario-folder', type=str, required=True,
              help='The scenario folder in Balmorel/ to run optimisations and simulations')
@click.option('--dark-style', is_flag=True, required=False, help='Dark plot style')
@click.option('--plot-ext', type=str, default='.pdf', required=False, help='The extension of the plot, defaults to ".pdf"')
@click.option('--pretrain-epochs', type=int, default=5, show_default=True,
              help='Number of epochs for initial pretraining')
@click.option('--update-epochs', type=int, default=1, show_default=True,
              help='Epochs used during each training update step')
@click.option('--days', type=int, default=1, show_default=True,
              help='Length of scenario blocks in days')
@click.option('--n-scenarios', type=int, default=2, show_default=True,
              help='How many scenarios to generate each round')
@click.option('--latent-dim', type=int, default=64, show_default=True,
              help='Latent dimension for the scenario generator')
@click.option('--seed', type=int, default=42, show_default=True,
              help='Random seed')
@click.option('--batch-size', type=int, default=256, show_default=True,
              help='Batch size used for pretraining / generation')
@click.option('--learning-rate', type=float, default=5e-4, show_default=True,
              help='Learning rate for the scenario generator (pretrain/update)')
def CLI(ctx, scenario_name: str, scenario_folder: str, dark_style: bool, plot_ext: str,
        pretrain_epochs: int, update_epochs: int, days: int, n_scenarios: int,
        latent_dim: int, seed: int, batch_size: int, learning_rate: float):
    """
    Online learning driver with logging to src/Logs/<scenario>_<timestamp>.log

    This CLI allows you to run online learning experiments with various configurations.

    Args:
        scenario_name (str): The name of the scenario to run.
        scenario_folder (str): The folder where the specific scenario will be run.
        days (int): The number of days to generate.
        n_scenarios (int): The number of scenarios (interpreted as weeks) to generate.
        pretrain_epochs (int): The number of pretraining epochs to run.
        dark_style (bool): Whether to use a dark plot style.
        plot_ext (str): The file extension for the plot output.
    """

    # Set global style of plot
    if dark_style:
        plt.style.use('dark_background')
        fc = 'none'
    else:
        fc = 'white'

    # Store global options in the context object
    ctx.ensure_object(dict)
    ctx.obj['fc'] = fc
    ctx.obj['plot_ext'] = plot_ext
    
    # logging
    logger, logfile = _setup_logger(scenario_name)
    logger.info("=== Online learning run started ===")
    logger.info(f"scenario={scenario_name} pretrain_epochs={pretrain_epochs} update_epochs={update_epochs} "
                f"days={days} n_scenarios={n_scenarios} latent_dim={latent_dim} seed={seed} "
                f"batch_size={batch_size} learning_rate={learning_rate}")
    logger.info(f"Logfile: {logfile}")

    # pretraining + initial scenario generation
    logger.info("Starting pretraining...")
    
    epoch = 0    
    model = pretrain(pretrain_epochs, days=days, n_scenarios=n_scenarios, latent_dim=latent_dim, batch_size=batch_size, learning_rate=learning_rate, seed=seed, logger=logger, n_features=80)
    

    logs_dir = logfile.parent
    ckpt_path = logs_dir / f"{scenario_name}_model_checkpoint.pth"

    os.chdir('Balmorel')
    sp.run(f'mkdir simex_{scenario_name}', shell=True)
    sp.run(f'mv {scenario_folder}/model/balopt.opt {scenario_folder}/model/balopt_dispatch.opt', shell=True) # make sure first run is a capacity expansion

    while epoch < update_epochs:
        for runtype in ['capacity', 'dispatch']:
            
            sp.run(f'rm {scenario_folder}/data/*.inc', shell=True)
            
            if runtype == 'capacity':
                sp.run(f'cp -f {scenario_folder}/capexp_data/*.inc {scenario_folder}/data', shell=True)
                sp.run(f'mv {scenario_folder}/data/DE.inc {scenario_folder}/data/ANTBALM_FICTDE.inc', shell=True)
                sp.run(f'mv {scenario_folder}/data/DH.inc {scenario_folder}/data/ANTBALM_FICTDH.inc', shell=True)
                sp.run(f'mv {scenario_folder}/data/HYDROGEN_DH2.inc {scenario_folder}/data/ANTBALM_FICTDH2.inc', shell=True)
            else:
                sp.run(f'cp -f {scenario_folder}/S.inc {scenario_folder}/data', shell=True)
                sp.run(f'cp -f {scenario_folder}/T_{runtype}.inc {scenario_folder}/data/T.inc', shell=True)
                sp.run(f'touch {scenario_folder}/data/ANTBALM_FICTDE.inc', shell=True)
                sp.run(f'touch {scenario_folder}/data/ANTBALM_FICTDH.inc', shell=True)
                sp.run(f'touch {scenario_folder}/data/ANTBALM_FICTDH2.inc', shell=True)
                sp.run(f'mv {scenario_folder}/model/balopt_{runtype}.opt {scenario_folder}/model/balopt.opt', shell=True)

            # Copy from the simex folder
            if epoch > 0 and runtype == 'dispatch':
                sp.run(f'cp -rf simex_{scenario_name}/* simex', shell=True)
            
            os.chdir(f'{scenario_folder}/model')
            sp.run(f'gams Balmorel --scenario_name "{scenario_name}_{runtype}_E{epoch}"', shell=True)

            # Check feasibility
            out = sp.run('cat Balmorel.lst | grep "LP status"', shell=True, capture_output=True).stdout
            if not 'optimal' in str(out):
                logger.warning(f"{scenario_name}_{runtype}_E{epoch} infeasible!")
            else:
                logger.info(f"{scenario_name}_{runtype}_E{epoch} feasible.")

            os.chdir('../../')

            # Copy the simex folder          
            sp.run(f'cp -rf simex/* simex_{scenario_name}', shell=True)

            if runtype != "capacity":
                # Rename balopt back
                sp.run(f'mv "{scenario_folder}/model/balopt.opt" "{scenario_folder}/model/balopt_{runtype}.opt"', shell=True)
        
        sp.run(f'pixi run python -u analysis/analyse.py adequacy "{scenario_name}_dispatch" {epoch}', shell=True)
        os.chdir('../')
        model = train(model, scenario_name, epoch, n_scenarios=n_scenarios, batch_size=batch_size, logger=logger, scenario_folder=scenario_folder)
        
        # save the model
        model.save_model(str(ckpt_path))
        logger.info(f"Epoch {epoch} completed and model saved.")
        
        os.chdir('Balmorel')
        
        epoch += 1

#%% ------------------------------- ###
###            2. Utils             ###
### ------------------------------- ###

@click.pass_context
def plot_style(ctx, fig, ax, name: str, legend: bool = True):
    
    ax.set_facecolor(ctx.obj['fc'])
    
    if legend:
        ax.legend(loc='center', bbox_to_anchor=(.5, 1.15), ncol=3)
    
    fig.savefig(name + ctx.obj['plot_ext'], bbox_inches='tight', transparent=True)
    
    return fig, ax

#%% ------------------------------- ###
###             3. Main             ###
### ------------------------------- ###
if __name__ == '__main__':
    CLI()
