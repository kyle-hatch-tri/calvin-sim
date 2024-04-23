import subprocess
import time 


command_str = """calvin
export AWS_ACCESS_KEY_ID="ASIARZ3C2ZCO3YA47CE3"
export AWS_SECRET_ACCESS_KEY="B9ZTUCK7cUJNBEmmwx4/FFwZIcon8XB6iHbTIAl9"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEAEaCXVzLWVhc3QtMSJHMEUCIQCx05xVpBfoIB4SCAybS9dyGEorkRtT0Z7lfkfJNzDVswIgEojuQWJN5+Gk/4sAiwleP4Z5OBDwtg/MGt8Snlnlbz8qiQMIOhAEGgwxMjQyMjQ0NTY4NjEiDEgDefp3njcFCxNuTSrmAr9b9Au/nUFNfnkfWR8uvbjAMOJUxADKhf5VSEmDjlJjvh8iA4iKIDy7kxHqg/uP3LGR8NsZjn2yYKFJMSsOomDYbEyt3dz6fu38wKycVSFPESMi9iPTaPCXTL1GGFAd1kucsS91Uqa3rXBAExFHao01WGKaoC1UGtEIaswhgtZ816xxgForV3fjuubOkP5fRr+lkAYiI9tIUfagEXLJ9mFpISJRhMm/PsRBYpTUG8iKwg0xT2Adzvz66eXHanxHvGOM3BS2UySwA+EVY/wb9kUuxlmuwTtoLlCsp0B+cv8sEHANytX7f+dA+DzDdFAeUUMW4/DE0cXphD9N96QRZH7bxhV8KvssiiR/084VwDi3qPxMQ2OY9NPwvHdVBDqCrPBukoeQ/ey2w+IfHE7SQ4+VOQKl/18lxHpUzHi6MbGPeSEAWNaIAnCLwvsvSfNSYfOzXfDOKQGVDrK0W3iJIhIiKtWIE7Aw/uLcsAY6pgHfaNdO5scUD0RkCRQuK81NczxG/edbwMNHZU08eFi92/yAfeT6xZQadUHNEIp8TfFw8JU9RixOJG+Gf4f6ZmN6a4l8xBAZIJCnUVHhyV0sPvPGnEo8UQhbe7pAN4caiOCygJj2M4jnosC6+sG7uW3BDBbSEFLCOhJfS6h5XXWXMbXk+zdsRO/diIXGiPeLaA47x2090ht9urujQHyqkQmUYAPKFtuI"

python3 -u sagemaker_launch_generated_goals.py \\
--user kylehatch \\
--base-job-name calvin-sim \\
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \\
--machine_start_idx {} \\
--machine_n_idxs {} \\
--n_samples {} \\
--s3_save_uri "s3://susie-data/calvin_data_processed/language_conditioned_{}_samples_encodedecode_noisedencodedecode" \\
--diffusion_model_checkpoint s3://kyle-sagemaker-training-outputs/susie_test/public_model/checkpoint_only/ \\
--input-source s3 \\
--instance_type ml.p4d.24xlarge 
"""

# s3://kyle-sagemaker-training-outputs/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/

def create_tmux_window(session_name, command):
    # Create a new tmux session with the specified name
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name])

    # # Create a new window within the session with the specified name
    # subprocess.run(['tmux', 'new-window', '-t', f'{session_name}:', '-n', window_name])

    # Send a command to the newly created window
    # subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}:{window_name}', f'{command}', 'Enter'])

    subprocess.run(['tmux', 'send-keys', '-t', f'{session_name}', f'{command}', 'Enter'])


def format_time(seconds):
    # Calculate days, hours, minutes, and remaining seconds
    days = seconds // (24 * 3600)
    seconds %= 24 * 3600
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    # Format the time string
    time_str = "{:02d}:{:02d}:{:02d}:{:02d}".format(days, hours, minutes, seconds)
    
    return time_str




FN_INPUTS = 24053
INPUTS_PER_MACHINE = 1600 
N_SAMPLES = 4
TIME_PER_N_SAMPLES = 9.468


# FN_INPUTS = 24053
# INPUTS_PER_MACHINE = 3200 
# N_SAMPLES = 16
# TIME_PER_N_SAMPLES = 9.468

# FN_INPUTS = 24053
# INPUTS_PER_MACHINE = 3200 
# N_SAMPLES = 8
# TIME_PER_N_SAMPLES = 6

assert INPUTS_PER_MACHINE % 8 == 0




n_commands = 0
for i in range(FN_INPUTS // INPUTS_PER_MACHINE):
    start_idx = i * INPUTS_PER_MACHINE 
    session_name = f"g{start_idx}"
    command = command_str.format(start_idx, INPUTS_PER_MACHINE, N_SAMPLES, N_SAMPLES)

    create_tmux_window(session_name, command)
    print(command)
    n_commands += 1

    time.sleep(0.1)

print("Number of sagemaker jobs:", n_commands)
n_per_gpu = INPUTS_PER_MACHINE // 8
time_per_job = int(n_per_gpu * 64 * TIME_PER_N_SAMPLES)
print(f"Run time per job ({n_per_gpu} trajectories of length 64 per gpu, {TIME_PER_N_SAMPLES}s to generate {N_SAMPLES} samples):", format_time(time_per_job))

