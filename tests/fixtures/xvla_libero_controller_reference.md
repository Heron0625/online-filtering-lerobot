Controller reference for X-VLA LIBERO target conversion

Source: local raw LIBERO `libero_goal/turn_on_the_stove_demo.hdf5`, demo_25,
90 non-noop commands. Generated on 2026-10-30 with the installed LIBERO and
robosuite environment, after the standard ten settling commands. At each step,
state contains current controller position and simulator **body** rotvec (not
cached observations); target comes directly from controller.set_goal(action[:6])
(goal_pos, goal_ori columns, binary gripper, unused second arm zeros).
initial_goal contains controller.goal_ori before the first command.
Five commands have zero rotation and must preserve the preceding goal.

Reproducer: `research/outputs/xvla_data_audit/verify_corrected_same_state.py`
(local diagnostic artifact; original data are not required to run the test).
