from io import StringIO
import logging
import os
from typing import Annotated, Optional

import numpy as np


import vtk
from vtkmodules.vtkCommonDataModel import (
    vtkIterativeClosestPointTransform,
    vtkPolyData,
)
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter


import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLModelNode, vtkMRMLTextNode


#
# MEP
#

class MEP(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MEP"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["NeuroMapping"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Lucas Betioli (USP); Lucas da Costa (USP)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#MEP">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # MEP1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='MEP',
        sampleName='MEP1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'MEP1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='MEP1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='MEP1'
    )

    # MEP2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='MEP',
        sampleName='MEP2',
        thumbnailFileName=os.path.join(iconsPath, 'MEP2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='MEP2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='MEP2'
    )


#
# MEPParameterNode
#

@parameterNodeWrapper
class MEPParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """
    inputModel: vtkMRMLModelNode
    inputMEP: vtkMRMLTextNode
    inputVolume: vtkMRMLScalarVolumeNode
    outputVolume: vtkMRMLScalarVolumeNode 
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# MEPWidget
#

class MEPWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/MEP.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MEPLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Set View
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputModel:
            firstModelNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode")
            if firstModelNode:
                self._parameterNode.inputModel = firstModelNode

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputMEP:
            firstTextNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLTextNode")
            if firstTextNode:
                self._parameterNode.inputMEP = firstTextNode

    def setParameterNode(self, inputParameterNode: Optional[MEPParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputModel and self._parameterNode.inputMEP and self._parameterNode.outputVolume:
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.mepSelector.currentNode())

#
# MEPLogic
#

class MEPLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return MEPParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLModelNode,
                outputVolume: vtkMRMLModelNode,
                mepModel: vtkMRMLTextNode,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume or not mepModel:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        '''
        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)
        '''

        threshold_down = 0

        dims_size = 1001

        icp = False

        gaussian_sharpness = 3
        gaussian_radius = 4

        def ICP(coord, surface):
            """
            Apply ICP transforms to fit the points to the surface
            Args:
                coord: raw coordinates to apply ICP
            """
            sourcePoints = np.array(coord[:3])
            sourcePoints_vtk = vtkPoints()
            for i in range(len(sourcePoints)):
                id0 = sourcePoints_vtk.InsertNextPoint(sourcePoints)
            source = vtkPolyData()
            source.SetPoints(sourcePoints_vtk)

            source_points = source

            icp = vtkIterativeClosestPointTransform()
            icp.SetSource(source_points)
            icp.SetTarget(surface)

            icp.GetLandmarkTransform().SetModeToRigidBody()
            # icp.GetLandmarkTransform().SetModeToAffine()
            icp.DebugOn()
            icp.SetMaximumNumberOfIterations(100)
            icp.Modified()
            icp.Update()

            icpTransformFilter = vtkTransformPolyDataFilter()
            icpTransformFilter.SetInputData(source_points)
            icpTransformFilter.SetTransform(icp)
            icpTransformFilter.Update()

            transformedSource = icpTransformFilter.GetOutput()
            p = [0, 0, 0]
            transformedSource.GetPoint(0, p)

            return p[0], p[1], p[2], None, None, None

        inputNode = inputVolume.GetStorageNode()

        stl_reader = vtk.vtkSTLReader()
        stl_reader.SetFileName(inputNode.GetFileName())
        stl_reader.Update()
        #surface = inputVolume.GetPolyData()
        surface = stl_reader.GetOutput()
        bounds = np.array(surface.GetBounds())

        text_data = mepModel.GetText()
        temp_file_path = 'temp_data.txt'
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(text_data)

        points_reader = vtk.vtkDelimitedTextReader()
        points_reader.SetFileName(temp_file_path)
        points_reader.DetectNumericColumnsOn()
        points_reader.SetFieldDelimiterCharacters('\t')
        points_reader.SetHaveHeaders(True)

        print(points_reader)

        # create the vtkTable object
        tab = vtk.vtkTable()
        table_points = vtk.vtkTableToPolyData()
        table_points.SetInputConnection(points_reader.GetOutputPort())
        table_points.SetXColumnIndex(0)
        table_points.SetYColumnIndex(1)
        table_points.SetZColumnIndex(2)
        table_points.Update()

        points = table_points.GetOutput()

        print(points)

        points.GetPointData().SetActiveScalars('MEP')
        print(points.GetPointData())
        range_up = points.GetPointData().GetScalars().GetRange()[1]
        rang = (threshold_down, range_up)

        dims = np.array([dims_size, dims_size, dims_size])
        box = vtk.vtkImageData()
        box.SetDimensions(dims)
        box.SetSpacing((bounds[1::2] - bounds[:-1:2])/(dims - 1))
        box.SetOrigin(bounds[::2])

        # Gaussian kernel
        gaussian_kernel = vtk.vtkGaussianKernel()
        gaussian_kernel.SetSharpness(gaussian_sharpness)
        gaussian_kernel.SetRadius(gaussian_radius)

        interpolator = vtk.vtkPointInterpolator()
        interpolator.SetInputData(box)
        interpolator.SetSourceData(points)
        interpolator.SetKernel(gaussian_kernel)

        resample = vtk.vtkResampleWithDataSet()
        resample.SetInputData(surface)
        resample.SetSourceConnection(interpolator.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(resample.GetOutputPort())
        mapper.SetScalarRange(rang)

        lut = vtk.vtkLookupTable()
        lut.SetTableRange(threshold_down, range_up)
        colorSeries = vtk.vtkColorSeries()
        seriesEnum = colorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_9
        colorSeries.SetColorScheme(seriesEnum)
        colorSeries.BuildLookupTable(lut, colorSeries.ORDINAL)
        lut_map = vtk.vtkLookupTable()
        lut_map.DeepCopy(lut)
        lut_map.Build()
        mapper.SetLookupTable(lut_map)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        point_mapper = vtk.vtkPointGaussianMapper()
        point_mapper.SetInputData(points)
        point_mapper.SetScalarRange(rang)
        point_mapper.SetScaleFactor(0.75)
        point_mapper.EmissiveOff()
        point_mapper.SetSplatShaderCode(
            "//VTK::Color::Impl\n"
            "float dist = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);\n"
            "if (dist > 1.0) {\n"
            "  discard;\n"
            "} else {\n"
            "  float scale = (1.0 - dist);\n"
            "  ambientColor *= scale;\n"
            "  diffuseColor *= scale;\n"
            "}\n"
        )

        point_mapper.SetLookupTable(lut)
        point_actor = vtk.vtkActor()
        point_actor.SetMapper(point_mapper)

        actor.SetOrientation((-12, 41, 66))
        point_actor.SetOrientation((-12, 41, 66))

        colorBarActor = vtk.vtkScalarBarActor()
        colorBarActor.SetLookupTable(lut)
        labelFormat = vtk.vtkTextProperty()

        colorBarActor.SetTitle("MEP amplitude µV\n")
        titleFormat = vtk.vtkTextProperty()
        titleFormat.SetVerticalJustificationToTop()
        colorBarActor.SetVisibility(1)


        def OnPressLeftButton(evt, obj):
            print(actor.GetOrientation())

        #renderer = vtk.vtkRenderer()
        #renWin = vtk.vtkRenderWindow()
        #renWin.AddRenderer(renderer)
        #renWin.SetSize(2048, 1080)
        #iren = vtk.vtkRenderWindowInteractor()
        #iren.SetRenderWindow(renWin)
        #renderer.AddActor(actor)
        #renderer.AddActor(point_actor)
        #renderer.AddActor(colorBarActor)
        #cam = renderer.GetActiveCamera()
        #renderer.ResetCamera()
        #print(cam.SetPosition((181.50680842279354, 97.51127257102047+200, 864.2770996170809)))
        #cam.Zoom(6)
        #iren.Initialize()
        #renWin.Render()
        #iren.Start()

        inputVolume.SetDisplayVisibility(False)
        slicer.util.resetThreeDViews()
        
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        renderWindow = view.renderWindow()
        renderers = renderWindow.GetRenderers()
        renderer = renderers.GetItemAsObject(0)
        renderer.AddActor(actor)
        renderer.AddActor(point_actor)
        renderer.AddActor(colorBarActor)
        slicer.util.forceRenderAllViews()

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# MEPTest
#

class MEPTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_MEP1()

    def test_MEP1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('MEP1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = MEPLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
