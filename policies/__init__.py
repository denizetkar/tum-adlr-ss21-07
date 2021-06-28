from stable_baselines3.common.policies import register_policy

from .actor_critic import ActorCriticCnnPolicy, ActorCriticCnnRnnPolicy, ActorCriticPolicy, ActorCriticRnnPolicy

__all__ = [
    "ActorCriticCnnPolicy",
    "ActorCriticPolicy",
    "ActorCriticRnnPolicy",
    "ActorCriticCnnRnnPolicy",
]

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("RnnPolicy", ActorCriticRnnPolicy)
register_policy("CnnRnnPolicy", ActorCriticCnnRnnPolicy)
