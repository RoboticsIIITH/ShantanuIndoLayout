    

def get_default_config(cfg_default):
    """Get default configuration from file"""
    config = load_class('get_cfg_defaults',
                         paths=[cfg_default.replace('/', '.')],
                         concat=False)()
    config.merge_from_list(['default', cfg_default])
    return config

def parse_train_config(cfg_default, cfg_file):
    """
    Parse model configuration for training
    Parameters
    ----------
    cfg_default : str
        Default **.py** configuration file
    cfg_file : str
        Configuration **.yaml** file to override the default parameters
    Returns
    -------
    config : CfgNode
        Parsed model configuration
    """
    # Loads default configuration
    config = get_default_config(cfg_default)
    # Merge configuration file
    config = merge_cfg_file(config, cfg_file)
    # Return prepared configuration
    return prepare_train_config(config)
