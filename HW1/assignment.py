import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os
import sys
from hand import *

# tuning handles, these are derived by measuring the images in library using the stat flag

# if a contour's largest 4 defects are on average longer than this, it is a splayed hand
DEFECT_LENGTH_THRESHOLD = 5289.70

# if a non-splay hand's overall eccentricity is less than this, it is a fist, otherwise it is a palm
ECCENTRICITY_THRESHOLD = 0.36

# these are breakpoint coordinates for the vertical and horizontal position of the hand
# the origin is the top left corner of the image

LEFT_THRESHOLD = 358
RIGHT_THRESHOLD = 638

TOP_THRESHOLD = 403
BOTTOM_THRESHOLD = 621

# constants and flags

LIBRARY_PATH_IN = os.path.join(os.getcwd(), "library\in")
LIBRARY_PATH_OUT = os.path.join(os.getcwd(), "library\out")
# define some file level flags, VERBOSE, STAT, and SAVE
VERBOSE = False
STAT = False
SAVE = False
LOCK = False


def main():
    # detect debug and stat flags, adjust behavior accordingly
    if "--verbose" in sys.argv:
        print("Running in debug mode")
        global VERBOSE
        VERBOSE = True

    if "--stat" in sys.argv:
        print("Gathering calibration statistics")
        global STAT
        STAT = True

    if "--save" in sys.argv:
        print("Saving images")
        global SAVE
        SAVE = True

    if "--lock" in sys.argv:
        print("Evaluating combo lock")
        global LOCK
        LOCK = True

    # variables for stat mode, unused otherwise
    if STAT:
        defect_lengths_anomaly = []
        defect_lengths_splay = []
        defect_lengths_fist = []
        defect_lengths_palm = []

        eccentricities_anomaly = []
        eccentricities_splay = []
        eccentricities_fist = []
        eccentricities_palm = []

        hpos_lefts = []
        hpos_centers = []
        hpos_rights = []

        vpos_tops = []
        vpos_centers = []
        vpos_bottoms = []

    # if in lock mode, evaluate the folder called lock in order of file name
    if LOCK:
        hands = []
        lockCombo = parse_combo_file(os.path.join("library", "lock", "combo.txt"))
        if VERBOSE:
            print(f"Combo: {lockCombo}")
        # if zero length, the combo file is empty
        if len(lockCombo) == 0:
            print("Combo file is empty, exiting")
            return
        lockLength = len(lockCombo)
        for i in range(lockLength):
            # check if file exists
            if not os.path.exists(os.path.join("library", "lock", f"{i}.jpeg")):
                print(f"File {i}.jpeg does not exist, exiting")
                return
            hands.append(
                reduce_to_features(os.path.join("library", "lock", f"{i}.jpeg"))
            )
            decide_pose(hands[i])
            decide_location(hands[i])
            if VERBOSE:
                print(hands[i])
        print("Hands read, evaluating...")
        success = evaluate_combo(hands, lockCombo)
        if VERBOSE:
            # show the images
            display_hand_sequence(lockCombo, hands, success)
        return
    # otherwise, walk through and analyze all the images in the library not in the lock folder
    else:
        for root, _, files in os.walk(LIBRARY_PATH_IN):
            for file in files:
                if file.endswith(".jpeg"):
                    print("=========================================\n")
                    # generate a preliminary hand object from the supplied filepath
                    hand = reduce_to_features(os.path.join(root, file))
                    # classify the hand
                    decide_pose(hand)
                    decide_location(hand)
                    # printing shows the GT and the classification
                    print(hand)

                    if VERBOSE:
                        print(hand.stat())

                # if in calibration mode, collect stats
                if STAT:
                    # classify by pose
                    if hand.hand_type == HandType.ANOMALY:
                        defect_lengths_anomaly.append(hand.data["defectTop4Avg"])
                        eccentricities_anomaly.append(hand.data["eccentricity"])
                    elif hand.hand_type == HandType.SPLAY:
                        defect_lengths_splay.append(hand.data["defectTop4Avg"])
                        eccentricities_splay.append(hand.data["eccentricity"])
                    elif hand.hand_type == HandType.FIST:
                        defect_lengths_fist.append(hand.data["defectTop4Avg"])
                        eccentricities_fist.append(hand.data["eccentricity"])
                    elif hand.hand_type == HandType.PALM:
                        defect_lengths_palm.append(hand.data["defectTop4Avg"])
                        eccentricities_palm.append(hand.data["eccentricity"])

                    # group by vertical position
                    if hand.vpos == HandVPos.TOP:
                        vpos_tops.append(hand.data["center"][1])
                    elif hand.vpos == HandVPos.CENTER:
                        vpos_centers.append(hand.data["center"][1])
                    elif hand.vpos == HandVPos.BOTTOM:
                        vpos_bottoms.append(hand.data["center"][1])

                    # group by horizontal position
                    if hand.hpos == HandHPos.LEFT:
                        hpos_lefts.append(hand.data["center"][0])
                    elif hand.hpos == HandHPos.CENTER:
                        hpos_centers.append(hand.data["center"][0])
                    elif hand.hpos == HandHPos.RIGHT:
                        hpos_rights.append(hand.data["center"][0])
                else:
                    # save hand images
                    save_hand_image(hand, LIBRARY_PATH_OUT)
                print("=========================================\n")
    # output the stats
    if STAT:
        # im sure you can loop or otherwise clean this up but its a one off solution and clarity is more important
        avg_defect_lengths_anomaly = sum(defect_lengths_anomaly) / len(
            defect_lengths_anomaly
        )
        avg_defect_lengths_splay = sum(defect_lengths_splay) / len(defect_lengths_splay)
        avg_defect_lengths_fist = sum(defect_lengths_fist) / len(defect_lengths_fist)
        avg_defect_lengths_palm = sum(defect_lengths_palm) / len(defect_lengths_palm)

        avg_eccentricities_anomaly = sum(eccentricities_anomaly) / len(
            eccentricities_anomaly
        )
        avg_eccentricities_splay = sum(eccentricities_splay) / len(eccentricities_splay)
        avg_eccentricities_fist = sum(eccentricities_fist) / len(eccentricities_fist)
        avg_eccentricities_palm = sum(eccentricities_palm) / len(eccentricities_palm)

        avg_hpos_left = sum(hpos_lefts) / len(hpos_lefts)
        avg_hpos_center = sum(hpos_centers) / len(hpos_centers)
        avg_hpos_right = sum(hpos_rights) / len(hpos_rights)

        avg_vpos_top = sum(vpos_tops) / len(vpos_tops)
        avg_vpos_center = sum(vpos_centers) / len(vpos_centers)
        avg_vpos_bottom = sum(vpos_bottoms) / len(vpos_bottoms)

        # pretty print the stats in a table using f strings, round to 2 decimal places
        print("=" * 65)
        print(
            f'{"Hand Type":<25}{"Average Defect Length":<25}{"Average Eccentricity":<25}'
        )
        print(
            f'{"Anomaly":<25}{avg_defect_lengths_anomaly:<25.2f}{avg_eccentricities_anomaly:<25.2f}'
        )
        print(
            f'{"Splay":<25}{avg_defect_lengths_splay:<25.2f}{avg_eccentricities_splay:<25.2f}'
        )
        print(
            f'{"Fist":<25}{avg_defect_lengths_fist:<25.2f}{avg_eccentricities_fist:<25.2f}'
        )
        print(
            f'{"Palm":<25}{avg_defect_lengths_palm:<25.2f}{avg_eccentricities_palm:<25.2f}'
        )
        print("=" * 65 + "\n")

        # do something similar for the horizontal and vertical positions
        print(f'{"Hand Position":<25}{"Average X":<25}{"Average Y":<25}')
        print(f'{"Left":<25}{avg_hpos_left:<25.2f}{"--":<25}')
        print(f'{"Center":<25}{avg_hpos_center:<25.2f}{"--":<25}')
        print(f'{"Right":<25}{avg_hpos_right:<25.2f}{"--":<25}')
        print(f'{"Top":<25}{"--":<25}{avg_vpos_top:<25.2f}')
        print(f'{"Center":<25}{"--":<25}{avg_vpos_center:<25.2f}')
        print(f'{"Bottom":<25}{"--":<25}{avg_vpos_bottom:<25.2f}')
        print("=" * 65 + "\n")

        suggested_defect_length_threshold = (
            avg_defect_lengths_splay
            + (avg_defect_lengths_fist + avg_defect_lengths_palm) / 2
        ) / 2
        suggested_eccentricity_threshold = (
            avg_eccentricities_fist + avg_eccentricities_palm
        ) / 2
        suggested_left_threshold = round((avg_hpos_left + avg_hpos_center) / 2)
        suggested_right_threshold = round((avg_hpos_right + avg_hpos_center) / 2)
        suggested_top_threshold = round((avg_vpos_top + avg_vpos_center) / 2)
        suggested_bottom_threshold = round((avg_vpos_bottom + avg_vpos_center) / 2)

        # average the fist and palm defect lengths together, then get their average with the splay defect length to get the threshold
        print(
            f'{"Pose Suggestions":<25}{"DEFECT_LENGTH_THRESHOLD":<25}{"OVAL_THRESHOLD":<25}'
        )
        print(
            f'{"":<25}{suggested_defect_length_threshold:<25.2f}{suggested_eccentricity_threshold:<25.2f}'
        )
        print("=" * 65 + "\n")
        # average together left and center hpos to get the left threshold, and center and right to get the right threshold
        print(
            f'{"Horizontal Suggestions":<25}{"LEFT THRESHOLD":<25}{"RIGHT THRESHOLD":<25}'
        )
        print(
            f'{"":<25}{suggested_left_threshold:<25.2f}{suggested_right_threshold:<25.2f}'
        )
        print("=" * 65 + "\n")
        # average together top and center vpos to get the top threshold, and center and bottom to get the bottom threshold
        print(
            f'{"Vertical Suggestions":<25}{"TOP THRESHOLD":<25}{"BOTTOM THRESHOLD":<25}'
        )
        print(
            f'{"":<25}{suggested_top_threshold:<25.2f}{suggested_bottom_threshold:<25.2f}'
        )

        # graph the defect lengths, eccentricities, and hand positions by axis
        # draw each threshold on the graph, only draw it at the necessary width between the two bars it is thresholding
        # space them out some
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.tight_layout(pad=3.0)
        fig.suptitle("Calibration Statistics")
        ax1.set_title("Defect Lengths")
        ax1.set_ylabel("Average Defect Length")
        ax1.set_xlabel("Hand Type")
        ax1.bar(
            ["Anomaly", "Splay", "Fist", "Palm"],
            [
                avg_defect_lengths_anomaly,
                avg_defect_lengths_splay,
                avg_defect_lengths_fist,
                avg_defect_lengths_palm,
            ],
        )
        ax1.hlines(
            y=suggested_defect_length_threshold,
            xmin=0.5,
            xmax=3.5,
            color="r",
            linestyle="-",
        )

        ax2.set_title("Eccentricities")
        ax2.set_ylabel("Average Eccentricity")
        ax2.set_xlabel("Hand Type")
        ax2.bar(
            ["Anomaly", "Splay", "Fist", "Palm"],
            [
                avg_eccentricities_anomaly,
                avg_eccentricities_splay,
                avg_eccentricities_fist,
                avg_eccentricities_palm,
            ],
        )
        ax2.hlines(
            y=suggested_eccentricity_threshold,
            xmin=1.5,
            xmax=3.5,
            color="r",
            linestyle="-",
        )

        ax3.set_title("Vertical Position")
        ax3.set_ylabel("Average Y Position")
        ax3.set_xlabel("Vertical Position")
        ax3.bar(
            ["Top", "Center", "Bottom"],
            [avg_vpos_top, avg_vpos_center, avg_vpos_bottom],
        )
        ax3.hlines(
            y=suggested_top_threshold, xmin=-0.5, xmax=1.5, color="r", linestyle="-"
        )
        ax3.hlines(
            y=suggested_bottom_threshold, xmin=0.5, xmax=2.5, color="r", linestyle="-"
        )

        ax4.set_title("Horizontal Position")
        ax4.set_ylabel("Average X Position")
        ax4.set_xlabel("Horizontal Position")
        ax4.bar(
            ["Left", "Center", "Right"],
            [avg_hpos_left, avg_hpos_center, avg_hpos_right],
        )
        ax4.hlines(
            y=suggested_left_threshold, xmin=-0.5, xmax=1.5, color="r", linestyle="-"
        )
        ax4.hlines(
            y=suggested_right_threshold, xmin=0.5, xmax=2.5, color="r", linestyle="-"
        )

        plt.show()
        # stats mode only, don't run the program
        return


# this function parses the combo.txt file to returns the lock combo
def parse_combo_file(path: str) -> list:
    print(f"Parsing combo file {path}...")
    lock = []
    with open(path, "r") as f:
        # parse the lock combo from the combo.txt file
        for line in f:
            line = line.strip()
            if line == "":
                continue
            line = line.split()
            lock.append((line[0], line[1], line[2]))
    return lock

# this function displays the hand sequence that was used to unlock the lock
def display_hand_sequence(combo, hands: list[Hand], wasSuccessful: bool):
    # display the thresholded hand sequence
    fig, axs = plt.subplots(1, len(hands))
    fig.tight_layout(pad=2.0)
    comboString = ""
    for hand in combo:
        comboString += f"{hand[0]} {hand[1]} {hand[2]}\n"
    if wasSuccessful:
        fig.suptitle(f"Successful Combo\n\n{comboString}")
    else:
        fig.suptitle(f"Failed Combo\n\n{comboString}")
    # hand.binary is a numpy array of the thresholded hand, show this as an image
    for i, hand in enumerate(hands):
        axs[i].set_title(f"Hand {i + 1} \n {hand.data['predictedHandType']} at {hand.data['predictedVPos']} {hand.data['predictedHPos']}")
        axs[i].imshow(hand.data['binary'], cmap="gray")
        # put WRONG below the image if the prediction was wrong or CORRECT if it was correct
        if combo[i][0] == "NONE":
            axs[i].text(0.5, 0.5, "CORRECT", horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, color="green")
        elif combo[i][0] != hand.data['predictedHandType'] or combo[i][1] != hand.data['predictedVPos'] or combo[i][2] != hand.data['predictedHPos']:
            axs[i].text(0.5, 0.5, "WRONG", horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, color="red")
        else:
            axs[i].text(0.5, 0.5, "CORRECT", horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, color="green")
    plt.show()
    
# this function evaluates the lock combo against the predictions and returns whether or not the combo was correct
def evaluate_combo(lock, hands: list[Hand]) -> bool:
    if len(lock) != len(hands):
        if VERBOSE:
            print(
                f"Lock length does not match gathered length. Lock length: {len(lock)}, Gathered length: {len(hands)}"
            )
        return False

    # check against the predictions, not ground truth
    for lock_phase, hand in zip(hands, lock):
        print(
            f'Checking lock phase {lock_phase} against prediction {hand.data["predictedHandType"]}, {hand.data["predictedVPos"]}, {hand.data["predictedHPos"]}'
        )
        # each prediction is a hand object
        # each lock_phase is a tuple of the hand type, vertical position, and horizontal position
        # "NONE" is a wildcard, so if a hand type, vertical position, or horizontal position is "NONE", it will match any hand type, vertical position, or horizontal position
        if lock_phase[0] != "NONE" and lock_phase[0] != hand.data["predictedHandType"]:
            print(
                f'Lock phase {lock_phase} does not match prediction {hand.data["predictedHandType"]} on hand \n {hand}'
            )
            return False
        if lock_phase[1] != "NONE" and lock_phase[1] != hand.data["predictedVPos"]:
            print(
                f'Lock phase {lock_phase} does not match vertical prediction {hand.data["predictedVPos"]} on hand \n {hand}'
            )
            return False
        if lock_phase[2] != "NONE" and lock_phase[2] != hand.data["predictedHPos"]:
            print(
                f'Lock phase {lock_phase} does not match horizontal prediction {hand.data["predictedHPos"]} on hand \n {hand}'
            )
            return False
        if VERBOSE:
            print(
                f'Lock phase {lock_phase} matches prediction {hand.data["predictedHandType"]}, {hand.data["predictedVPos"]}, {hand.data["predictedHPos"]}'
            )
    print("Lock combination is correct! Opening...")
    return True

# this function is the first phase of reduction, it generates contours and binary images
def reduce_to_binary(image: cv.Mat) -> tuple:
    # pull out the red channel
    red_channel = image[:, :, 2]

    # gaussian blur the image to improve thresholding
    blurred = cv.GaussianBlur(red_channel, (11, 11), 0)

    # otsu thresholding to get a binary image
    _, thresholded_image = cv.threshold(
        blurred, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )

    # remove some noise through morphological operations
    kernel = np.ones((15, 15), np.uint8)
    thresholded_image = cv.morphologyEx(thresholded_image, cv.MORPH_OPEN, kernel)

    # fill in gaps through morphological operations
    thresholded_image = cv.morphologyEx(thresholded_image, cv.MORPH_CLOSE, kernel)

    # update the binary image in the hand object

    # get the largest contour and remove the rest, final denoising step
    contours, _ = cv.findContours(
        thresholded_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    hand_contour = max(contours, key=lambda x: cv.contourArea(x))

    # get contour indices of the convex hull
    hull = cv.convexHull(hand_contour, returnPoints=False)

    # return a dict of "hull" and "contour"
    return hull, contours, hand_contour, thresholded_image


# this function uses the binary image to get the a handful of features that will be used to classify the hand
def reduce_to_features(path) -> Hand:
    print("Analyzing hand...")
    # initialize the hand object
    hand = Hand(path)

    # reduce the image and update the hand object
    (
        hand.data["hull"],
        hand.data["fullContour"],
        hand.data["handContour"],
        hand.data["binary"],
    ) = reduce_to_binary(hand.img)

    # select the hullth points of the contour to make other functions work and set them aside
    contour_of_hull = hand.data["handContour"][hand.data["hull"][:, 0]]

    # get the convexity defects from the contour and hull
    defects = cv.convexityDefects(hand.data["handContour"], hand.data["hull"])

    # splayed hands have 4 large convexity defects, non-splayed hands do not, this can be used to distinguish between the two
    hand.data["defectTop4Avg"] = defects[:, 0, 3].mean()

    ellipse = cv.fitEllipse(contour_of_hull)

    # get the eccentricity of the contour
    # when calculating it this way, 0 is a line, 1 is a circle, and 0.5 is an ellipse
    hand.data["eccentricity"] = 1 - (ellipse[1][0] / ellipse[1][1])

    # store the geometric center of the hull in pixels
    M = cv.moments(contour_of_hull)
    hand.data["center"] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # return the hand with the data filled in
    return hand


# this function shows several images of the hand in different stages of processing
def display_hand(hand):
    # title the entire plot with the prediction
    fig, axs = plt.subplots(1, 4)
    fig.suptitle(
        f"Analysis for Prediction: {hand.data['predictedHandType']} at {hand.data['predictedVPos']} {hand.data['predictedHPos']}"
    )
    # space the subplots out, put the title close directly above the graphs
    fig.tight_layout(pad=0.5)
    # make the image wider than it is tall
    fig.set_size_inches(12, 4)
    # draw the contour on the original image, titled "original"
    axs[0].set_title("Original")
    axs[0].imshow(
        cv.drawContours(
            cv.cvtColor(hand.img, cv.COLOR_BGR2RGB),
            hand.data["fullContour"],
            -1,
            (0, 255, 0),
            5,
        )
    )
    # show the binary image
    axs[1].set_title("Binary")
    axs[1].imshow(hand.data["binary"], cmap="gray")

    # connect the dots on the hull and draw it on a copy of the image
    # use modulus to wrap around to the first point
    hull_image = hand.img.copy()

    for i in range(len(hand.data["hull"])):
        cv.line(
            hull_image,
            tuple(hand.data["handContour"][hand.data["hull"][i][0]][0]),
            tuple(
                hand.data["handContour"][
                    hand.data["hull"][(i + 1) % len(hand.data["hull"])][0]
                ][0]
            ),
            (0, 255, 0),
            5,
        )
    axs[2].set_title("Hull")
    axs[2].imshow(cv.cvtColor(hull_image, cv.COLOR_BGR2RGB))

    image_with_defects = hand.img.copy()
    contour_of_hull = hand.data["handContour"][hand.data["hull"][:, 0]]
    # strip to the 4 largest defects
    defects = cv.convexityDefects(hand.data["handContour"], hand.data["hull"])
    defects = defects[np.argsort(defects[:, 0, 3])][::-1][:4]
    # draw each defect as a triangle between the start, end, and far points
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(hand.data["handContour"][s][0])
        end = tuple(hand.data["handContour"][e][0])
        far = tuple(hand.data["handContour"][f][0])
        cv.line(image_with_defects, start, end, [0, 0, 255], 2)
        cv.line(image_with_defects, start, far, [0, 0, 255], 2)
        cv.line(image_with_defects, end, far, [0, 0, 255], 2)

    # draw the geometric center of the convex hull in a thick yellow circle
    cx = hand.data["center"][0]
    cy = hand.data["center"][1]
    cv.circle(image_with_defects, (cx, cy), 5, (0, 255, 255), -1)

    # show the fitted ellipse
    ellipse = cv.fitEllipse(contour_of_hull)
    cv.ellipse(image_with_defects, ellipse, (0, 255, 0), 5)
    axs[3].set_title("Defects")
    axs[3].imshow(cv.cvtColor(image_with_defects, cv.COLOR_BGR2RGB))
    if VERBOSE:
        plt.show()
    if VERBOSE:
        # in a separate image, show the original image with the prediction labels on it
        # extend the image down a few hundred pixels to show the prediction
        prediction_image = np.zeros(
            (hand.img.shape[0] + 100, hand.img.shape[1], 3), np.uint8
        )
        prediction_image[: hand.img.shape[0], : hand.img.shape[1]] = hand.img
        # draw the predictedHandType, predictedVPos, and predictedHPos on the image on different lines
        cv.putText(
            prediction_image,
            "Predicted Hand Type: " + hand.data["predictedHandType"],
            (10, hand.img.shape[0] + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            prediction_image,
            "Predicted Vertical Position: " + hand.data["predictedVPos"],
            (10, hand.img.shape[0] + 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv.LINE_AA,
        )
        cv.putText(
            prediction_image,
            "Predicted Horizontal Position: " + hand.data["predictedHPos"],
            (10, hand.img.shape[0] + 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
            cv.LINE_AA,
        )
        # resize the image down some
        prediction_image = cv.resize(prediction_image, (0, 0), fx=0.5, fy=0.5)

        cv.imshow("Prediction", prediction_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return fig


# save annotated or intermediate hand to this path
def save_hand_image(hand: Hand, path: str):
    print("Saving image...")
    subfolder = hand.hand_type.name.lower()
    fig = display_hand(hand)
    # save the figure as a png and then convert it to a jpeg, required since matplotlib doesn't save as jpeg
    fig.savefig(f"{path}\\{subfolder}\\INTER_{hand.originalName}")
    plt.close(fig)
    prediction_image = np.zeros(
        (hand.img.shape[0] + 100, hand.img.shape[1], 3), np.uint8
    )
    prediction_image[: hand.img.shape[0], : hand.img.shape[1]] = hand.img

    # draw the predictedHandType, predictedVPos, and predictedHPos on the image on different lines at font scale 1
    cv.putText(
        prediction_image,
        "Predicted Hand Type: " + hand.data["predictedHandType"],
        (10, hand.img.shape[0] + 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        prediction_image,
        "Predicted Vertical Position: " + hand.data["predictedVPos"],
        (10, hand.img.shape[0] + 60),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        prediction_image,
        "Predicted Horizontal Position: " + hand.data["predictedHPos"],
        (10, hand.img.shape[0] + 90),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )
    # resize the image down some
    prediction_image = cv.resize(prediction_image, (0, 0), fx=0.9, fy=0.9)
    # save the image
    cv.imwrite(f"{path}\\{subfolder}\\PRED_{hand.originalName}.jpeg", prediction_image)


# this function takes a hand and decides what pose it is in, then stores the result in the hand's data
def decide_pose(hand: Hand) -> str:
    print("Deciding pose...")
    # anomalies have both a high eccentricity and a high average of the top 4 convexity defects from the statistical analysis
    # TODO: anomalies are likely to pick a strange hue of red to threshold at, maybe that's a good way to detect them?

    if (hand.data["eccentricity"] > ECCENTRICITY_THRESHOLD) and (
        hand.data["defectTop4Avg"] > DEFECT_LENGTH_THRESHOLD
    ):
        if VERBOSE:
            print("Anomaly detected")
            print(str(hand.data["eccentricity"]) + " > " + str(ECCENTRICITY_THRESHOLD))
            print(
                str(hand.data["defectTop4Avg"]) + " > " + str(DEFECT_LENGTH_THRESHOLD)
            )
        hand.data["predictedHandType"] = HandType.ANOMALY.name
        return

    # splays have high average of the top 4 convexity defects from the statistical analysis
    # fists and palms have low average of the top 4 convexity defects from the statistical analysis
    # fists have a low eccentricity and palms have a high eccentricity

    elif hand.data["defectTop4Avg"] >= DEFECT_LENGTH_THRESHOLD:
        if VERBOSE:
            print("Splay detected")
            print(
                str(hand.data["defectTop4Avg"]) + " < " + str(DEFECT_LENGTH_THRESHOLD)
            )
        hand.data["predictedHandType"] = HandType.SPLAY.name
        return

    elif hand.data["eccentricity"] >= ECCENTRICITY_THRESHOLD:
        if VERBOSE:
            print("Palm detected")
            print(str(hand.data["eccentricity"]) + " > " + str(ECCENTRICITY_THRESHOLD))
        hand.data["predictedHandType"] = HandType.PALM.name
        return

    else:
        hand.data["predictedHandType"] = HandType.FIST.name
        if VERBOSE:
            print("Fist detected")
            print(str(hand.data["eccentricity"]) + " > " + str(ECCENTRICITY_THRESHOLD))
        return


# this function takes a hand and decides where it is in the picture, then stores the result in the hand's data
def decide_location(hand: Hand) -> tuple:
    print("Deciding location...")
    # if anomaly, return
    if hand.data["predictedHandType"] == HandType.ANOMALY.name:
        return

    x, y = hand.data["center"]
    predictedHPos = None
    predictedVPos = None

    if x < LEFT_THRESHOLD:
        predictedHPos = HandHPos.LEFT.name
    elif x > RIGHT_THRESHOLD:
        predictedHPos = HandHPos.RIGHT.name
    else:
        predictedHPos = HandHPos.CENTER.name

    # this is backwards because the origin is at the top left
    if y < TOP_THRESHOLD:
        predictedVPos = HandVPos.TOP.name
    elif y > BOTTOM_THRESHOLD:
        predictedVPos = HandVPos.BOTTOM.name
    else:
        predictedVPos = HandVPos.CENTER.name

    hand.data["predictedVPos"] = predictedVPos
    hand.data["predictedHPos"] = predictedHPos
    if VERBOSE:
        print(f"Predicted position: {predictedVPos} {predictedHPos}")

    return


if __name__ == "__main__":
    main()
