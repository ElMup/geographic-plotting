import xml.etree.ElementTree as ET
from matplotlib import colors
import os


def read_colors_from_xml(xml_path):
    """Function name says it al

    Parameters
    ----------
    xml_path

    Returns
    -------

    """
    qml = ET.parse(xml_path).getroot()
    colorlist = [colors.to_rgba("#ffffff")] * (int(qml[2][0][2][0].attrib["value"]) - 1)
    for colorentry in qml[2][0][2]:
        colorlist.append(colors.to_rgba(colorentry.attrib["color"]))
    return colorlist


def read_labels_from_xml(xml_path):
    """Function name says it al

    Parameters
    ----------
    xml_path

    Returns
    -------

    """
    qml = ET.parse(xml_path).getroot()
    colorlist = [""] * (int(qml[2][0][2][0].attrib["value"]) - 1)
    for colorentry in qml[2][0][2]:
        colorlist.append(colorentry.attrib["label"])
    return colorlist


def qml_harmonizer(path_leader, paths_followers):
    leader = ET.parse(path_leader)
    leader_palette = leader.getroot()[2][0][2]

    for path in paths_followers:
        follower = ET.parse(path)
        follower_palette = follower.getroot()[2][0][2]

        for i, leader_entry in enumerate(leader_palette):
            label = leader_entry.attrib["label"][4:]
            color = leader_entry.attrib["color"]

            isin = False
            for follower_entry in follower_palette:
                if follower_entry.attrib["label"][4:] == label:
                    follower_entry.attrib["color"] = color
                    isin = True
            if not isin:
                print(f"{label} is not in {path}")

        follower.write(f"{os.path.dirname(path)}/mt_map.xml")
