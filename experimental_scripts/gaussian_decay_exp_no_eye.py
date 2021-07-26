import os, sys, random, datetime, operator
from psychopy import visual, core, event, monitors, info, gui
from pandas import read_csv, DataFrame
from psychopy.tools.colorspacetools import rgb2dklCart
import numpy as np
from random import shuffle

import pandas as pd

import warnings

warnings.simplefilter('ignore')



current_time = datetime.datetime.today().strftime("%m%d%Y_%H%M%S")

user_input_dict = {
    "CoAx ID [#]": "",
    "Session [##] (1-100)": "",
    "Run [#] (1-2)": "",
}
sub_inf_dlg = gui.DlgFromDict(
    user_input_dict,
    title="Subject information",
    show=0,
    order=[
        "CoAx ID [#]",
        "Session [##] (1-100)",
        "Run [#] (1-2)",
    ],
)

# set data path & collect information from experimenter
testing = int(input("Testing? "))

if testing !=  1 and testing != 0:
    sys.exit("Enter 0 or 1.")

# parent_directory = '/Users/coax_lab/Desktop/diffusive_bandit'
parent_directory = os.path.join(os.path.expanduser('~'), 'Desktop/diffusive_bandit')

image_directory = parent_directory + "/images/"
exp_param_directory = parent_directory + "/experimental_parameters/reward_parameters/"
data_directory = parent_directory + "/data/BIDS/"
run_info_directory = parent_directory + "/data/run_info_data/"

# deterministic_exp_param_directory = os.getcwd() + '/experimental_parameters/deterministic_schedules/'

sub_inf_dlg.show()
subj_id = int(float(user_input_dict["CoAx ID [#]"]))
session_n = int(float(user_input_dict["Session [##] (1-100)"]))
run = int(float(user_input_dict["Run [#] (1-2)"]))


subj_directory = data_directory + "sub-" + "{:02d}".format(subj_id) + "/"
session_directory = subj_directory + "ses-" + "{:02d}".format(session_n) + "/"

behavioral_directory = session_directory + "beh/"
func_directory = session_directory + "func/"

directories = list([behavioral_directory, func_directory, run_info_directory])

for dir in directories:
    if not os.path.exists(dir):
        os.makedirs(dir)

exp_param_file = (
    exp_param_directory
    + 'sub' + str(subj_id)
    + '_ses' + str(session_n)
    + "_diffusive_bandit_"
    + "run"
    + str(run)
    + ".csv"
)
print("EXP PARAM" + exp_param_file)
if not os.path.exists(exp_param_file):
    sys.exit("Experimental parameter file does not exist.")

output_file_name = (
    "sub-"
    + "{:02d}".format(subj_id)
    + "_"
    + "ses-"
    + "{:02d}".format(session_n)
    + "_"
    + "task-"
    + "diffusive-bandit_"
    + "run-"
    + "{:02d}".format(run)
    + "_"
    + str(current_time)
)

if testing:
    output_file_name = output_file_name + '_test'

# error checking for user input

n_runs = 2

print(str(subj_id), str(run), str(session_n))

try:

    assert len(str(run)) == 1
    assert run > 0 & run < (n_runs + 1)

except AssertionError:
    sys.exit(
        "Format failure. Run number should be 1:2."
    )


output_path_beh = behavioral_directory + output_file_name + ".json"
output_path_readable_beh = behavioral_directory + output_file_name + ".csv"

run_info_path = run_info_directory + output_file_name + "_runInfo.csv"
output_path_events = func_directory + output_file_name + "_events.tsv"


if not testing and os.path.exists(output_path_readable_beh):
    sys.exit(output_file_name + " already exists! Overwrite danger...Exiting.")

exp_param = read_csv(exp_param_file, header=0)
exp_param.columns = exp_param.columns.str.strip()

reward_f = np.round(exp_param.reward_f.values, 2).astype("int")
reward_m = np.round(exp_param.reward_m.values, 2).astype("int")


rewards = np.transpose(np.array([reward_f, reward_m]))
max_reward_idx = np.argmax(rewards, 1)
min_reward_idx = np.argmin(rewards, 1)
n_trials = len(exp_param.trial)

f_image = exp_param.f_image.tolist()
m_image = exp_param.m_image.tolist()


run_test_trials = 80


n_test_trials = run_test_trials  # needs to be divisible by 2

if testing:
    n_trials = n_test_trials


print("n_trials: ", n_trials)

trial_list = list(np.arange(0, n_trials))

total_reward = 0 # start with 0
response_failure_reward = -20

vertical_txt_break = "\n" * 10
small_vertical_txt_break = "\n" * 2
horiz_txt_break = "\t" * 5

instructions_p1 = (
    "You are going on a treasure hunt! You will start with "
    + str(total_reward)
    + " coins, and you will be able to pay a coin "
    + "to ask one of two greebles if they have money. On each trial you will meet two greebles: one is female, one is male."
    + small_vertical_txt_break
    + "This is a female."
    + horiz_txt_break
    + "This is a male."
    + vertical_txt_break
    + "Note how their features differ. The female greeble has a downward facing appendage, whereas the male greeble has an upward facing appendage.\n\nPress the left button when you're ready to continue to the next instruction screen.  "
)

instructions_p2 = (
    "On each trial you can ask either the male or female greeble for money. If the greeble you ask chooses to give you money, "
    + "he or she will give you a certain number of coins to add to your bank.\n\nSometimes females will give coins more often. Sometimes males will give coins more often. "
    + "Your goal is simply to make as much money as possible by learning which type of greeble will give you more money. "
)

instructions_p3 = (
    "The total amount of money that you have is shown as a bank at the center of the screen:"
    + small_vertical_txt_break * 7
    + "If you earn money, the bank will turn green. "
    + "If you lose money, the bank will turn yellow.\n\nIf you choose too slowly or too quickly, you will lose 5 points and the bank at the center of the screen will turn red.\n\nEach point you earn will correspond to one real cent that you will be paid in addition to your hourly pay. So do the best you can!"
)


instructions_p4 = (
    "To ask the left greeble for money, press the left button with your left thumb. "
    + "To ask the right greeble for money press the right button with your right thumb.\n\n"
    + "Between trials, please focus your eyes on the bank.\n\nPress the left button when you are ready to begin the hunt!"
)


instructions_p5 = (
    "Do your best to focus on the bank. Press the left button to start the task."
)


slow_trial = "Too slow! \nChoose quickly."
fast_trial = "Too fast! \nSlow down."
between_run_inst = (
    "Feel free to take a break! \nPress the left button when you're ready to continue."
)


# initialize dependent variables
rt_list = [np.nan] * n_trials
identity_choice_list = [np.nan] * n_trials
LR_solution_list = [np.nan] * n_trials
LR_choice_list = [np.nan] * n_trials
value_accuracy_list = [np.nan] * n_trials
point_value_list = [np.nan] * n_trials

subj_id_list = [subj_id] * n_trials
run_list = [run] * n_trials

# instantiate psychopy object instances
expTime_clock = core.Clock()
trialTime_clock = core.Clock()
rt_clock = core.Clock()

screen_size = (1920., 1200.)  # screen size in pixels
window_size = (1280., 800.)
mon = monitors.Monitor(
    "BOLD_display", width=20.7, distance=56,
)  # width and distance in cm
mon.setSizePix(screen_size)
mon.saveMon()


center = (0,0)

luminance = 10
contrast = 1

dkl_purple = (luminance, 300, contrast)
dkl_red = (luminance, 45, contrast)
dkl_gray = (luminance, 0, 0)
dkl_green = (luminance, 145, contrast)
dkl_orange = (luminance, 45, contrast)
dkl_yellow = (luminance, 80, contrast)

dkl_blue = (luminance, 225, contrast)


greeble_color = dkl_purple
inst_color = [1, 1, 1]
speed_message_color = dkl_red

window = visual.Window(
    size=screen_size,
    units="pix",
    monitor=mon,
    color=dkl_blue,
    colorSpace="dkl",
    blendMode="avg",
    useFBO=True,
    allowGUI=False,
    fullscr=False,
    pos=center,
    screen=1,
)

break_msg = visual.TextStim(
    win=window,
    units="pix",
    antialias="False",
    text=between_run_inst,
    wrapWidth=window_size[0] - 400,
    height=window_size[1] / 32,
)
inst_msg = visual.TextStim(
    win=window,
    units="pix",
    antialias="False",
    colorSpace="dkl",
    color=[90, 0, 1],
    wrapWidth=window_size[0] - 400,
    height=window_size[1] / 28,
)
end_msg = visual.TextStim(
    win=window,
    units="pix",
    antialias="False",
    wrapWidth=window_size[0] - 400,
    colorSpace="dkl",
    color=[90, 0, 1],
    height=window_size[1] / 32,
)
speed_msg = visual.TextStim(
    win=window,
    units="pix",
    antialias="False",
    text=slow_trial,
    wrapWidth=window_size[0] - 400,
    height=window_size[1] / 15,
    alignHoriz="center",
    colorSpace="rgb",
    color=[1, -1, -1],
    bold=True,
)

# m/f from different families to emphasize dimension of interest (sex)
female_greeble_sample = visual.ImageStim(
    window,
    image=image_directory + "symm_greebles/" + "f1~11-v1.tif",
    units="pix",
    size=[window_size[0] / 5],
    colorSpace="dkl",
    color=greeble_color,
)
male_greeble_sample = visual.ImageStim(
    window,
    image=image_directory + "symm_greebles/" + "m2~21-v1.tif",
    units="pix",
    size=[window_size[0] / 5],
    colorSpace="dkl",
    color=greeble_color,
)

# take in an image list
female_greeble = visual.ImageStim(
    window,
    image=image_directory + "symm_greebles/" + "f1~11-v1.tif",
    units="pix",
    size=[window_size[0] / 4],
    colorSpace="dkl",
    color=greeble_color,
)
male_greeble = visual.ImageStim(
    window,
    image=image_directory + "symm_greebles/" + "m1~11-v1.tif",
    units="pix",
    size=[window_size[0] / 4],
    colorSpace="dkl",
    color=greeble_color,
)

runtimeInfo = info.RunTimeInfo(
    author="kb", win=window, userProcsDetailed=False, verbose=True
)
fixation_point_reward_total = visual.TextStim(
    win=window,
    units="pix",
    antialias="False",
    pos=[0, 15],
    colorSpace="dkl",
    color=dkl_gray,
    height=window_size[0] / 20,
)

cost_per_decision = -1

cue_list = [female_greeble, male_greeble]

high_val_cue = []
low_val_cue = []

# define target coordinates
left_pos_x = -window_size[0] / 5
right_pos_x = window_size[0] / 5


n_reps = n_trials // 2
l_x = np.tile(left_pos_x, n_reps)
r_x = np.tile(right_pos_x, n_reps)
l_r_x_arr = np.concatenate((l_x, r_x))



# shuffle target coordinates
np.random.seed()
np.random.shuffle(l_r_x_arr)


rt_max = 0.75
rt_min = 0.1

mandatory_trial_time = 1.5



left_key = "2"
right_key = "1"
inst_key = left_key

escape_key = "escape"


severe_error_color = dkl_red  # SEVERE error: no response or too fast. -x points.
error_color = dkl_yellow  # SEVERE error: no response or too fast. -x points.

neutral_color = dkl_gray  # no change
good_color = dkl_green  # earned points


# just record identity of male/female greeble with highest value

# initalize lists
received_rewards = []
total_rewards = []
value_correct_choices = []


# timing lists
stim_onset_list = []
stim_offset_list = []
trial_onset_list = []
abs_response_time_list = []

trial_time = []

iti_list = exp_param.ITI.values[:n_trials].tolist()
m_f_points = exp_param[['reward_f', 'reward_m']]
high_value_identity = exp_param.high_value_identity.values.tolist()


# high_val_cue_list = exp_param.p_id_solution[0:n_trials].tolist()
f_images = exp_param.f_image[:n_trials].tolist()
m_images = exp_param.m_image[:n_trials].tolist()


m_image_list = [
    image_directory + "symm_greebles/" + str(m_image) for m_image in m_images
]
f_image_list = [
    image_directory + "symm_greebles/" + str(f_image) for f_image in f_images
]


# give instructions
instruction_phase = True
while instruction_phase:
    inst_msg.text = instructions_p1
    inst_msg.setAutoDraw(True)
    female_greeble_sample.setPos([-200, 0])
    male_greeble_sample.setPos([200, 0])
    female_greeble_sample.draw()
    male_greeble_sample.draw()
    window.flip()
    inst_keys_p1 = event.waitKeys(keyList=[inst_key, escape_key])
    if escape_key in inst_keys_p1:
        sys.exit("escape key pressed.")

    inst_msg.text = instructions_p2
    window.flip()
    inst_keys_p2 = event.waitKeys(keyList=[inst_key, escape_key])
    if escape_key in inst_keys_p2:
        sys.exit("escape key pressed.")

    inst_msg.text = instructions_p3
    female_greeble_sample.setPos([-200, 75])
    male_greeble_sample.setPos([200, 75])
    female_greeble_sample.draw()
    male_greeble_sample.draw()
    fixation_point_reward_total.text = str(total_reward)
    fixation_point_reward_total.setPos([0, 75])
    fixation_point_reward_total.draw()
    window.flip()
    inst_keys_p3 = event.waitKeys(keyList=[inst_key, escape_key])
    if escape_key in inst_keys_p3:
        sys.exit("escape key pressed.")

    inst_msg.text = instructions_p4
    window.flip()
    inst_keys_p4 = event.waitKeys(keyList=[inst_key, escape_key])
    if escape_key in inst_keys_p4:
        sys.exit("escape key pressed.")

    inst_msg.text = instructions_p5
    window.flip()
    inst_keys_p4 = event.waitKeys(keyList=[inst_key, escape_key])
    if escape_key in inst_keys_p4:
        sys.exit("escape key pressed.")
    instruction_phase = False

inst_msg.setAutoDraw(False)
window.flip()


trigger = "5"
trigger_wait_instructions = "Waiting for trigger from the scanner..."


# test the trigger
inst_msg.text = trigger_wait_instructions
inst_msg.setAutoDraw(True)
window.flip()
print("Waiting for trigger...")
trigger_output = event.waitKeys(keyList=[trigger], clearEvents=True)

start_time = expTime_clock.getTime() # experiment starts now

inst_msg.setAutoDraw(False)
window.flip()


t = 0


expTime_clock.reset()  # reset so that inst. time is not included
trialTime_clock.reset()
fixation_point_reward_total.text = str(total_reward)
fixation_point_reward_total.setPos([0, 15])


# present choices
while t < n_trials:

    # trial has started, get time
    trial_start = expTime_clock.getTime() - start_time

    trial_onset_list.append(trial_start)

    trialTime_clock.reset()  # reset time

    fixation_point_reward_total.setAutoDraw(True)

    female_greeble.setPos([l_r_x_arr[t], 15])
    male_greeble.setPos([-l_r_x_arr[t], 15])

    female_greeble.setImage(f_image_list[t])
    male_greeble.setImage(m_image_list[t])

    cue_list[0].setAutoDraw(True)
    cue_list[1].setAutoDraw(True)
    window.flip()

    stim_onset_time = expTime_clock.getTime()

    stim_onset_list.append(stim_onset_time)

    rt_clock.reset()
    response = event.waitKeys(
        keyList=[left_key, right_key, escape_key], clearEvents=True, maxWait=rt_max
    )

    abs_response_time = expTime_clock.getTime()
    abs_response_time_list.append(abs_response_time)

    if response is None:
        rt = np.nan  # no response
        choice = np.nan
        identity_choice_list[t] = (np.nan)
    else:
        rt = rt_clock.getTime()
        choice = response[0][0]
        if escape_key in response:
            sys.exit("escape key pressed.")

    if cue_list[0] == female_greeble:
        left_key_points = m_f_points.reward_f[t]
        right_key_points = m_f_points.reward_m[t]
        left_identity, right_identity = 'f', 'm'

    elif cue_list[0] == male_greeble:
        left_key_points = m_f_points.reward_m[t]
        right_key_points = m_f_points.reward_f[t]
        left_identity, right_identity = 'm', 'f'

    if left_identity == high_value_identity[t]:
        LR_solution = 'L'
    elif right_identity == high_value_identity[t]:
        LR_solution = 'R'


    print('CHOICE ', choice)

    if choice == left_key:


        value_accuracy = left_identity == high_value_identity[t]
        value_accuracy_list[t] = (value_accuracy)

        LR_choice_list[t] = ord("L")
        identity_choice_list[t] = left_identity

        point_value = left_key_points

    elif choice == right_key:

        value_accuracy = right_identity == high_value_identity[t]
        value_accuracy_list[t] = value_accuracy

        LR_choice_list[t] = ord("R")
        identity_choice_list[t] = right_identity

        point_value = right_key_points

    elif np.isnan(choice):
        LR_choice_list[t] = (np.nan)
        identity_choice_list[t] = (np.nan)
        value_accuracy_list[t] = (np.nan)

        point_value = np.nan

    if point_value < 0:
        fixation_point_reward_total.color = error_color

    if rt < rt_max and rt > rt_min:
        received_rewards.append(point_value)
        total_reward += point_value
        fixation_point_reward_total.color = good_color
        window.flip()

    elif rt >= rt_max or rt <= rt_min:
        received_rewards.append(0)
        total_reward += response_failure_reward
        fixation_point_reward_total.color = severe_error_color
        window.flip()

    elif np.isnan(rt):
        received_rewards.append(0)
        total_reward += response_failure_reward
        fixation_point_reward_total.color = severe_error_color
        window.flip()

    fixation_point_reward_total.text = str("{:,}".format(total_reward))
    window.flip()

    point_value_list[t] = (point_value)
    total_rewards.append(total_reward)
    rt_list[t] = rt
    LR_solution_list[t] = LR_solution


    core.wait(
        mandatory_trial_time - trialTime_clock.getTime()
    )  # wait until mandatory trial time has passed

    cue_list[0].setAutoDraw(False)
    cue_list[1].setAutoDraw(False)

    trial_time.append(
        trialTime_clock.getTime()
    )  # trial time will always be set, sanity check

    # jitter iti & continue to show bank as fixation point
    fixation_point_reward_total.color = neutral_color
    window.flip()
    stim_offset_time = expTime_clock.getTime()

    stim_offset_list.append(stim_offset_time)

    core.wait(iti_list[t])
    response = event.getKeys(keyList=[escape_key])
    if escape_key in response:
        sys.exit()

    window.flip()

    t += 1

fixation_point_reward_total.setAutoDraw(False)
total_exp_time = expTime_clock.getTime() - start_time
stimulus_duration_list = list(map(operator.sub, stim_offset_list, stim_onset_list))


data_dict = {'trial': trial_list, 'subj_id': subj_id_list, 'run': run_list,
'LR_choice': LR_choice_list, 'identity_choice': identity_choice_list,
'value_accuracy': value_accuracy_list, 'LR_solution': LR_solution_list,
 'reward':received_rewards, 'cumulative_reward': total_rewards, 'rt': rt_list,
  'total_trial_time': trial_time,
'iti': iti_list, 'high_value_identity': high_value_identity,
 'id_choice': identity_choice_list, 'stim_duration': stimulus_duration_list,
  'stim_onset': stim_onset_list, 'stim_offset': stim_offset_list,
   'abs_response_time': abs_response_time_list,
    'reward_f': reward_f, 'reward_m': reward_m}

data = pd.DataFrame(data_dict)

runtime_data = np.matrix(
    (
        str(runtimeInfo["psychopyVersion"]),
        str(runtimeInfo["pythonVersion"]),
        str(runtimeInfo["pythonScipyVersion"]),
        str(runtimeInfo["pythonPygletVersion"]),
        str(runtimeInfo["pythonPygameVersion"]),
        str(runtimeInfo["pythonNumpyVersion"]),
        str(runtimeInfo["pythonWxVersion"]),
        str(runtimeInfo["windowRefreshTimeAvg_ms"]),
        str(runtimeInfo["experimentRunTime"]),
        str(runtimeInfo["experimentScript.directory"]),
        str(runtimeInfo["systemRebooted"]),
        str(runtimeInfo["systemPlatform"]),
        str(runtimeInfo["systemHaveInternetAccess"]),
        total_exp_time,
    )
)

runtime_header = "psychopy_version, python_version, pythonScipyVersion,\
pyglet_version, pygame_version, numpy_version, wx_version, window_refresh_time_avg_ms,\
begin_time, exp_dir, last_sys_reboot, system_platform, internet_access,\
 total_exp_time"

run_end_msg_text = (
    "Awesome! You have "
    + fixation_point_reward_total.text
    + " coins.\nYou have reached the end of the run.\nPlease wait for the experimenter to continue."
)
# dismiss participant
end_msg.text = run_end_msg_text
end_msg.draw()
core.wait(2)
window.flip()


events_header = ("stim_onset, stim_duration, trial_type, rt, accuracy")
events_data = np.transpose(
    np.matrix(
        (
            stim_onset_list[:t],
            stimulus_duration_list[:t],
            trial_list[:t],
            rt_list[:t],
            value_accuracy_list[:t],

        )
    )
)
np.savetxt(
    output_path_events, events_data, header=events_header, delimiter="\t", comments="")

print('events data saved')


data.to_csv(output_path_beh, index=False)
data.to_csv(output_path_readable_beh, index=False)

np.savetxt(
    run_info_path,
    runtime_data,
    header=runtime_header,
    delimiter=",",
    comments="", fmt="%s")

print(output_path_readable_beh, run_info_path)

response = event.waitKeys(keyList=[escape_key])


if escape_key in response:
    window.close()
    core.quit()
