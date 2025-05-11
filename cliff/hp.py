
from typing     import NamedTuple
from pathlib    import Path
import hashlib, json

class Params(NamedTuple):
    total_episodes: int      # Total episodes
    learning_rate: float     # Learning rate
    gamma: float             # Discounting rate
    epsilon: float           # Exploration probability
    min_epsilon: float
    map_size: int            # Number of tiles of one side of the squared environment
    seed: int                # Define a seed so that we get reproducible results
    is_slippery: bool        # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int              # Number of runs
    action_size: int         # Number of possible actions
    state_size: int          # Number of possible states
    proba_frozen: float      # Probability that a tile is frozen
    savefig_folder: str     # Root folder where plots are saved
    env_name: str
    render_mode: dict
    max_episode_steps_: int 
    limit_of_stubbornness: int
    n_views: int
    act: list
    starting_state:int



def get_config_hash(config: Params) -> str:
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj.resolve())  # full path
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return tuple(convert(x) for x in obj)
        elif isinstance(obj, float):
            return round(obj, 10)  # standardize float precision
        else:
            return obj

    # Convert the config to a serializable form
    data = convert(config._asdict())

    # Dump to JSON string (ensures consistent ordering), then hash
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode()).hexdigest()

