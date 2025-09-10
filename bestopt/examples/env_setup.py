from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment

cm = ConfigurationManager("config_setup.json")
env = BESTOptEnvironment(cm.config)

system_state = env.reset()



# # Simple simulation loop
# for step in range(96):  # 24 hours with 15-min resolution
#
#     # Create system action (normally this would come from controllers)
#     system_action = create_sample_action(system_state)
#
#     # Step the environment
#     system_state, done, info = env.step(system_action)
#
#     # Log progress
#     if step % 24 == 0:  # Every 6 hours
#         print_step_summary(step, system_state, info)
#
#     if done:
#         break
#
# # 4. Final summary
# print("\nSimulation completed!")
# final_summary = env.get_system_summary()
# print(f"Final system summary: {final_summary}")