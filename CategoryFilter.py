def filter_type_object_detection_sun_rgbd(word):

    # filters for bathtub
    if word == "tub":
        return "bathtub"

    # filters for bed
    if word == "bunk_bed":
        return "bed"

    # filters for shelf
    if word == "bookcase":
        return "bookshelf"


    # filters for box


    # filters for chair
    if word == "rocking_chair":
        return "chair"
    if word == "lawn_chair":
        return "chair"
    if word == "stack_of_chairs":
        return "chair"
    if word == "chairs":
        return "chair"
    if word == "high_chair":
        return "chair"
    if word == "lounge_chair":
        return "chair"
    if word == "chair_thumb":
        return "chair"
    if word == "rocking_chair":
        return "chair"
    if word == "lawn_chair":
        return "chair"
    if word == "stacked chairs":
        return "chair"

    # counter
    if word == "kitchen island":
        return "counter"

    # desk

    # door
    if word == "cdoor":
        return "door"

    # dresser

    # garbage-bin
    if word == "garbage_bin ":
        return "garbage-bin"
    if word == "garbage_bin":
        return "garbage-bin"
    if word == "recycle_bin":
        return "garbage-bin"

    # lamp
    if word == "walllamp":
        return "lamp"

    # monitor

    # night-stand
    if word == "nightstand":
        return "night-stand"
    if word == "night_stand":
        return "night-stand"

    # pillow


    # sink
    if word == "kitchen_sink":
        return "sink"

    # sofa
    if word == "sofa_chair":
        return "sofa"
    if word == "sofa_bed":
        return "sofa"
    if word == "ottoman":
        return "sofa"

    # table
    if word == "coffee_table":
        return "table"
    if word == "endtable":
        return "table"
    if word == "entable":
        return "table"
    if word == "dining table":
        return "table"
    if word == "sidetable":
        return "table"
    if word == "end_table":
        return "table"
    if word == "side_table":
        return "table"
    if word == "coffeetable":
        return "table"
    if word == "centertable":
        return "table"
    if word == "ping_pong_table":
        return "table"
    if word == "pingpongtable":
        return "table"
    if word == "long_office_table":
        return "table"
    if word == "bar_table":
        return "table"


    # television
    if word == "tv":
        return "television"
    if word == "flat_screen_tv":
        return "television"

    # toilet

    return word


def filter_type_instance_segmentation_sun_rgbd(word):
    # bathtub
    if word == "bathtab":
        return "bathtub"
    if word == "bath_dub":
        return "bathtub"

    # bed
    if word == "bunk bed":
        return "bed"



    # chair
    if word == "stacked chairs":
        return "chair"
    if word == "plastic chair":
        return "chair"
    if word == "sofa_chair":
        return "chair"


    # table
    if word == "game table":
        return "table"
    if word == "ping pong table":
        return "table"
    if word == "coffee table":
        return "table"


    # monitor
    if word == "computer_monitor":
        return "monitor"
    if word == "tv":
        return "monitor"
    if word == "television":
        return "monitor"


    if word == "garbage bin":
        return "garbage-bin"

    # pillow

    # sofa
    if word == "beige_sofa":
        return "sofa"

    # toilet

    # desk

    return word