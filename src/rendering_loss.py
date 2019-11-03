
from lxml import etree
from random import shuffle
import os
import numpy as np


def readInputXML(inputPath, shuffle):
    exampleDict = {}
    pathDict = {}
    tree = etree.parse(inputPath)
    for elem in tree.findall('.//item'):
        imagePath = elem.find('image').text
        if not (imagePath is None) and os.path.exists(imagePath):
            lightPower = elem.find('lightPower').text
            lightXPos = elem.find('lightXPos').text
            lightYPos = elem.find('lightYPos').text
            lightZPos = elem.find('lightZPos').text
            camXPos = elem.find('camXPos').text
            camYPos = elem.find('camYPos').text
            camZPos = elem.find('camZPos').text
            uvscale = elem.find('uvscale').text
            uoffset = elem.find('uoffset').text
            voffset = elem.find('voffset').text
            rotation = elem.find('rotation').text
            identifier = elem.find('identifier').text

            substanceName = imagePath.split("/")[-1]
            if (substanceName.split('.')[0].isdigit()):
                substanceName = '%04d' % int(substanceName.split('.')[0])
            substanceNumber = 0
            imageSplitsemi = imagePath.split(";")
            if len(imageSplitsemi) > 1:
                substanceName = imageSplitsemi[1]
                substanceNumber = imageSplitsemi[2].split(".")[0]

            material = inputMaterial(substanceName, lightPower, lightXPos, lightYPos, lightZPos, camXPos, camYPos,
                                     camZPos, uvscale, uoffset, voffset, rotation, identifier, imagePath)
            idkey = str(substanceNumber) + ";" + identifier.rsplit(";", 1)[0]

            if not (substanceName in exampleDict):
                exampleDict[substanceName] = {idkey: [material]}
                pathDict[imagePath] = material

            else:
                if not (idkey in exampleDict[substanceName]):
                    exampleDict[substanceName][idkey] = [material]
                    pathDict[imagePath] = material

                else:
                    exampleDict[substanceName][idkey].append(material)
    flatPathList = createMaterialTable(exampleDict, shuffle)
    return flatPathList


def createMaterialTable(examplesDict, shuffleImages):
    materialsList = []
    pathsList = []
    flatPathsList = []
    examplesDictKeys = examplesDict.keys()
    examplesDictKeys = sorted(examplesDict)

    for substanceName in examplesDictKeys:
        for variationKey, variationList in examplesDict[substanceName].items():
            materialsList.append(variationList)
            tmpPathList = []
            if a.mode == "test":
                for variation in variationList:
                    tmpPathList.append(variation.path)
            else:
                if len(variationList) > 1:
                    randomChoices = np.random.choice(variationList, 2, replace=False)
                    tmpPathList.append(randomChoices[0].path)
                    tmpPathList.append(randomChoices[1].path)
                else:
                    tmpPathList.append(variationList[0].path)
            pathsList.append(tmpPathList)
    if shuffleImages == True:
        shuffle(pathsList)

    for elem in pathsList:
        flatPathsList.extend(elem)
    return flatPathsList

class inputMaterial:
    def __init__(self, name, lightPower, lightXPos, lightYPos, lightZPos, camXPos, camYPos, camZPos, uvscale, uoffset, voffset, rotation, identifier, path):
        self.substanceName = name
        self.lightPower = lightPower
        self.lightXPos = lightXPos
        self.lightYPos = lightYPos
        self.lightZPos = lightZPos
        self.camXPos = camXPos
        self.camYPos = camYPos
        self.camZPos = camZPos
        self.uvscale = uvscale
        self.uoffset = uoffset
        self.voffset = voffset
        self.rotation = rotation
        self.identifier = identifier
        self.path = path

