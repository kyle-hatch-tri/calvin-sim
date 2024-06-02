import os 
import numpy as np 
import cv2

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
import libero.libero.utils.utils as libero_utils

import h5py


"""
suite=libero_data_processed_split2
split=val
a=$(aws s3 ls s3://susie-data/$suite/$split/libero_spatial/ | wc -l)
b=$(aws s3 ls s3://susie-data/$suite/$split/libero_object/ | wc -l)
c=$(aws s3 ls s3://susie-data/$suite/$split/libero_goal/ | wc -l)
d=$(aws s3 ls s3://susie-data/$suite/$split/libero_10/ | wc -l)
e=$(aws s3 ls s3://susie-data/$suite/$split/libero_90/ | wc -l)
sum=$((a + b + c + d + e))
echo $sum
"""


LIBERO_SPLIT1_TASKS = {
    "train": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "open_the_middle_drawer_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_the_stove",
        "put_the_bowl_on_top_of_the_cabinet",
        "put_the_cream_cheese_in_the_bowl",
        "put_the_wine_bottle_on_the_rack",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "turn_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
        "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE6_close_the_microwave",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
        "KITCHEN_SCENE7_open_the_microwave",
        "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
        "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
        "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
        "KITCHEN_SCENE8_turn_off_the_stove",
        "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
        "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE9_turn_on_the_stove",
        "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
        "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",

    ],

    "val": [
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
    ],
}

LIBERO_SPLIT2_TASKS = {
    "train": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "open_the_middle_drawer_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_the_stove",
        "put_the_bowl_on_top_of_the_cabinet",
        "put_the_cream_cheese_in_the_bowl",
        "put_the_wine_bottle_on_the_rack",
        "turn_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
        "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
        "KITCHEN_SCENE7_open_the_microwave",
        "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
        "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
        "KITCHEN_SCENE8_turn_off_the_stove",
        "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE9_turn_on_the_stove",
        "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
        "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",

    ],

    "val": [
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE6_close_the_microwave",
        "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
        "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
        "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    ],
}



libero_90_tasks = [
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
    "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
    "KITCHEN_SCENE3_turn_on_the_stove",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
    "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE6_close_the_microwave",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
    "KITCHEN_SCENE7_open_the_microwave",
    "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
    "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
    "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
    "KITCHEN_SCENE8_turn_off_the_stove",
    "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
    "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
    "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
    "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE9_turn_on_the_stove",
    "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
    "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
    "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
    "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
]

libero_goal_tasks = [
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
]

libero_object_tasks = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
]

libero_spatial_tasks = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
]

libero_10_tasks = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
]


# ls -1 * | sed 's/.*/"&",/'
LIBERO_SPLIT3_TASKS = {
    "train": libero_90_tasks + libero_goal_tasks + libero_object_tasks + libero_spatial_tasks,
    "val": libero_10_tasks,
}

LIBERO_SPLIT3_VAL90_TASKS = {
    "train": libero_90_tasks + libero_goal_tasks + libero_object_tasks + libero_spatial_tasks,
    "val": libero_90_tasks,
}

LIBERO_SPLIT4_TASKS = {
    "train": libero_90_tasks + libero_10_tasks + libero_goal_tasks + libero_object_tasks + libero_spatial_tasks,
    "val": libero_10_tasks,
}


LIBERO_SPLIT_ATM_LIBERO10_TASKS = {
    "train": libero_10_tasks,
    "val": libero_10_tasks,
}

LIBERO_SPLIT_ATM_LIBERO90_TASKS = {
    "train": libero_90_tasks,
    "val": libero_90_tasks,
}


LIBERO_SPLIT_ATM_LIBEROGOAL_TASKS = {
    "train": libero_goal_tasks,
    "val": libero_goal_tasks,
}


LIBERO_SPLIT_ATM_LIBEROOBJECT_TASKS = {
    "train": libero_object_tasks,
    "val": libero_object_tasks,
}

LIBERO_SPLIT_ATM_LIBEROSPATIAL_TASKS = {
    "train": libero_spatial_tasks,
    "val": libero_spatial_tasks,
}


def get_suite_from_task_name(task_name):
    if task_name in libero_10_tasks:
        return "libero_10"
    elif task_name in libero_spatial_tasks:
        return "libero_spatial"
    elif task_name in libero_object_tasks:
        return "libero_object"
    elif task_name in libero_goal_tasks:
        return "libero_goal"
    elif task_name in libero_90_tasks:
        return "libero_90"
    else:
        raise ValueError(f"Task name \"{task_name}\" not in any of the task suites.")







LIBERO_TASK_SPLITS = dict(liberosplit1=LIBERO_SPLIT1_TASKS,
                          liberosplit2=LIBERO_SPLIT2_TASKS,
                          liberosplit210shot=LIBERO_SPLIT2_TASKS,
                          liberosplit3=LIBERO_SPLIT3_TASKS,
                          liberosplit3val90=LIBERO_SPLIT3_VAL90_TASKS,
                          liberosplit4=LIBERO_SPLIT4_TASKS,
                          liberosplitatmlibero10=LIBERO_SPLIT_ATM_LIBERO10_TASKS,
                          liberosplitatmlibero90=LIBERO_SPLIT_ATM_LIBERO90_TASKS,
                          liberosplitatmliberogoal=LIBERO_SPLIT_ATM_LIBEROGOAL_TASKS,
                          liberosplitatmliberoobject=LIBERO_SPLIT_ATM_LIBEROOBJECT_TASKS,
                          liberosplitatmliberospatial=LIBERO_SPLIT_ATM_LIBEROSPATIAL_TASKS,)



RAW_DATA_FILES_DIR = "/home/kylehatch/Desktop/hidql/data/libero_data"

def get_suite_from_task(task_name):
    for suite_name, task_names in libero_task_map.items():
        if task_name in task_names:
            return suite_name 
        
    raise ValueError(f"Task \"{task_name}\" not found in libero_task_map")

def create_env(task_name, seed=None):
    benchmark_dict = benchmark.get_benchmark_dict()

    task_suite_name = get_suite_from_task(task_name)

    # task_suite_name = "libero_10"
    # task_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"

    task_suite = benchmark_dict[task_suite_name]()
    # retrieve a specific task
    task_names = [task.name for task in task_suite.tasks]

    assert task_name in task_names, f"\"{task_name}\" not in task_names: {task_names}"

        
    task_id = task_names.index(task_name)
    task = task_suite.get_task(task_id)
    assert task.name == task_name, f"task.name: {task.name} != task_name: {task_name}"

    # init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states

    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")


    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
        "control_freq":20,
    }  


    env = OffScreenRenderEnv(**env_args)

    if seed is not None:
        env.seed(seed)
        
    return env, task
        


class LiberoEnv:
    def __init__(self, env_name, im_size=(200, 200)) -> None:
        self.env_name = env_name
        self.task_names = LIBERO_TASK_SPLITS[env_name]
        self.im_size = im_size
        self.env = None

    @property
    def val_tasks(self):
        return self.task_names["val"]
    
    @property
    def train_tasks(self):
        return self.task_names["train"]
    
    @property
    def current_task(self):
        return self.task_name
    
    @property
    def language_instruction(self):
        return self.task.language
    
    def close_env(self):
        if self.env is not None:
            self.env.close()

    def reset(self, task_name, init_demo_id=None):
        self.close_env() # close existing env if one is open

        self.task_name = task_name
        self.env, self.task = create_env(task_name)


        

        reset_success = False
        while not reset_success:
            try:
                raw_obs = self.env.reset()
                reset_success = True
            except:
                continue

        if init_demo_id is not None:
            task_suite_name = get_suite_from_task_name(task_name) 

            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[task_suite_name]()
            task_names = [task.name for task in task_suite.tasks]

            task_id = task_names.index(self.task_name)
            task = task_suite.get_task(task_id)
            assert task.name == self.task_name, f"task.name: {task.name} != self.task_name: {self.task_name}"

            # init_states = task_suite.get_task_init_states(task_id)
            # self.env.set_init_state(init_states[init_demo_id])

            hdf5_file_path = os.path.join(RAW_DATA_FILES_DIR, task_suite_name, task_name + "_demo.hdf5")

            demo = f"demo_{init_demo_id}"
            with h5py.File(hdf5_file_path, 'r') as f:
                model_xml = f["data/{}".format(demo)].attrs["model_file"]
                model_xml = libero_utils.postprocess_model_xml(model_xml, {})

                # load the flattened mujoco states
                states = f["data/{}/states".format(demo)][()]

                init_idx = 0
                model_xml = model_xml.replace("/home/yifengz/workspace/libero-dev/chiliocosm", "/home/kylehatch/Desktop/hidql/calvin-sim/external/LIBERO/libero/libero")
                model_xml = model_xml.replace("/Users/yifengz/workspace/libero-dev/chiliocosm", "/home/kylehatch/Desktop/hidql/calvin-sim/external/LIBERO/libero/libero")
                
                self.env.reset_from_xml_string(model_xml)
                self.env.sim.reset()
                self.env.sim.set_state_from_flattened(states[init_idx])

                self.env.sim.forward()
            
        

        # Without this, some of the objects start off in the air and then fall to the table
        dummy_action = [0.] * 7
        for i in range(10):
            raw_obs, _, _, _ = self.env.step(dummy_action)

        return self.process_obs(raw_obs)
    
    def process_obs(self, raw_obs):
        img = raw_obs["agentview_image"]
        img = np.flip(img, axis=0)
        img = cv2.resize(img, self.im_size)

        return {"rgb_obs":{"rgb_static":img}}
    

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.process_obs(raw_obs)
        return obs, reward, done, info




# model.step(obs, lang_annotation)
        # rgb_obs = obs["rgb_obs"]["rgb_static"]



if __name__ == "__main__":
    env = LiberoEnv("liberosplit1")
    eval_tasks = env.val_tasks
    obs = env.reset(eval_tasks[0])
    rgb_obs = obs["rgb_obs"]["rgb_static"]

    dummy_action = [0.] * 7
    step = 0
    done = False 
    while not done:
        print("step:", step)
        obs, reward, done, info = env.step(dummy_action)
        step += 1

    import ipdb; ipdb.set_trace()