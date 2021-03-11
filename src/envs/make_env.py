import envs
def make(env, *args):
    if env == "Minecraft":
        env = envs.minecraft_RM.Minecraft(args)
    return env
