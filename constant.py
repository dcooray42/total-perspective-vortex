exp_1 = {"events" : {
            "T0" : 0
        },
        "event_id" : {
            "rest_eyes_open" : 0
        }
}
exp_2 = {"events" : {
            "T0" : 1
        },
        "event_id" : {
            "rest_eyes_close" : 1
        }
}
exp_3 = {"events" : {
            "T0" : 10,
            "T1" : 2,
            "T2" : 3
        },
        "event_id" : {
            "rest" : 10,
            "do_open_close_left_fist" : 2,
            "do_open_close_right_fist" : 3
        }
}
exp_4 = {"events" : {
            "T0" : 10,
            "T1" : 4,
            "T2" : 5
        },
        "event_id" : {
            "rest" : 10,
            "imagine_open_close_left_fist" : 4,
            "imagine_open_close_right_fist" : 5
        }
}
exp_5 = {"events" : {
            "T0" : 10,
            "T1" : 6,
            "T2" : 7
        },
        "event_id" : {
            "rest" : 10,
            "do_open_close_both_fist" : 6,
            "do_open_close_both_feet" : 7
        }
}
exp_6 = {"events" : {
            "T0" : 10,
            "T1" : 8,
            "T2" : 9
        },
        "event_id" : {
            "rest" : 10,
            "imagine_open_close_both_fist" : 8,
            "imagine_open_close_both_feet" : 9
        }
}
exp_sub_89 = {
    1 : {
            "events" : {
                "T0" : 0,
                "T1" : 0
            },
            "event_id" : {
            "rest_eyes_open" : 0
            }
        },
    2 : {
            "events" : {
                "T0" : 1,
                "T1" : 1
        },
        "event_id" : {
            "rest_eyes_open" : 1
        }
}
}

experiment_run = {
    1 : exp_1,
    2 : exp_2,
    3 : exp_3,
    4 : exp_4,
    5 : exp_5,
    6 : exp_6,
    7 : exp_3,
    8 : exp_4,
    9 : exp_5,
    10 : exp_6,
    11 : exp_3,
    12 : exp_4,
    13 : exp_5,
    14 : exp_6
}

tasks = {
    1 : [2, 6, 10],
    2 : [3, 7, 11],
    3 : [4, 8, 12],
    4 : [5, 9, 13]
}

tasks_labels = {
    1 : list(exp_3["event_id"].values()),
    2 : list(exp_4["event_id"].values()),
    3 : list(exp_5["event_id"].values()),
    4 : list(exp_6["event_id"].values())
}