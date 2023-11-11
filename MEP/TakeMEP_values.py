import vtk
import vtkmodules.all
import numpy as np
import scipy.io
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricTorus
from vtkmodules.vtkCommonCore import mutable, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellLocator,
    vtkIterativeClosestPointTransform,
    vtkPolyData,
)
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkPolyDataNormals, vtkAppendPolyData
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import (
    vtkArrowSource,
    vtkCylinderSource,
    vtkParametricFunctionSource,
    vtkRegularPolygonSource,
    vtkSphereSource,
)
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballActor,
    vtkInteractorStyleTrackballCamera,
)
from vtkmodules.vtkIOGeometry import vtkOBJReader, vtkSTLReader
from vtkmodules.vtkIOPLY import vtkPLYReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkFollower,
    vtkPolyDataMapper,
    vtkProperty,
    vtkRenderer,
)
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkWindowToImageFilter
)

from vtkmodules.vtkRenderingFreeType import vtkVectorText
from vtkmodules.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor
#import time

#SETTINGS
threshold_down = 0
#threshold_up = 500

dims_size = 1001
#dims_size = 501
icp = False
# offset_y = 20
# offset_z = 8
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

# Read a probe surface
stl_reader = vtk.vtkSTLReader()
stl_reader.SetFileName(r"C:\Users\Lucas biomag\Downloads\T1.stl")
#stl_reader = vtk.vtkOBJImporter()
#stl_reader.SetFileName('peel_brain.obj')
#stl_reader.SetFileNameMTL('peel_brain.mtl')
#stl_reader.SetFileName('peel_brain.stl')
stl_reader.Update()

surface = stl_reader.GetOutput()
bounds = np.array(surface.GetBounds())


data_filename=r"C:\Users\Lucas biomag\Documents\JoonasJoonas-MEP_raw.txt"
points_reader = vtk.vtkDelimitedTextReader()
points_reader.SetFileName(data_filename)
points_reader.DetectNumericColumnsOn()
points_reader.SetFieldDelimiterCharacters('\t')
points_reader.SetHaveHeaders(True)

# create the vtkTable object
tab = vtk.vtkTable()
table_points = vtk.vtkTableToPolyData()
table_points.SetInputConnection(points_reader.GetOutputPort())
table_points.SetXColumnIndex(0)
table_points.SetYColumnIndex(1)
table_points.SetZColumnIndex(2)
table_points.Update()

points = table_points.GetOutput()

points.GetPointData().SetActiveScalars('MEP')
range_up = points.GetPointData().GetScalars().GetRange()[1]
range = (threshold_down, range_up)

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
mapper.SetScalarRange(range)

lut = vtk.vtkLookupTable()
lut.SetTableRange(threshold_down, range_up)
colorSeries = vtk.vtkColorSeries()
seriesEnum = colorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_9
colorSeries.SetColorScheme(seriesEnum)
colorSeries.BuildLookupTable(lut, colorSeries.ORDINAL)
lut_map = vtk.vtkLookupTable()
lut_map.DeepCopy(lut)
#lut_map.SetTableValue(0, 1., 1., 1., 0.)
lut_map.Build()
mapper.SetLookupTable(lut_map)
#mapper.GetLookupTable().SetNanColor(0., 0., 0., 0.5)
actor = vtk.vtkActor()
actor.SetMapper(mapper)
#actor.GetProperty().SetInterpolationToGouraud()
point_mapper = vtk.vtkPointGaussianMapper()
point_mapper.SetInputData(points)
point_mapper.SetScalarRange(range)
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
#colorBarActor.SetOrientationToHorizontal()
#            self.colorBarActor.SetMaximumWidthInPixels( 50 )
colorBarActor.SetLookupTable(lut)
#colorBarActor.SetNumberOfLabels(2)
#colorBarActor.SetPosition(0.05, 0.1)
#colorBarActor.SetWidth(0.4)
#colorBarActor.SetHeight(0.08)
labelFormat = vtk.vtkTextProperty()
#labelFormat.SetFontSize(160)

colorBarActor.SetTitle("MEP amplitude µV\n")
titleFormat = vtk.vtkTextProperty()
#titleFormat.SetFontSize(250)
titleFormat.SetVerticalJustificationToTop()
#titleFormat.BoldOn()
#colorBarActor.SetPosition( pos[0], pos[1] )
#colorBarActor.SetLabelTextProperty(labelFormat)
#colorBarActor.SetTitleTextProperty(titleFormat)
colorBarActor.SetVisibility(1)
#colorBarActor.SetMaximumWidthInPixels(75)

def OnPressLeftButton(evt, obj):
    print(actor.GetOrientation())


renderer = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(2048, 1080)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)


#actor.GetProperty().SetOpacity(0.99)
renderer.AddActor(actor)
renderer.AddActor(point_actor)
renderer.AddActor(colorBarActor)

cam = renderer.GetActiveCamera()
renderer.ResetCamera()
print(cam.SetPosition((181.50680842279354, 97.51127257102047+200, 864.2770996170809)))
cam.Zoom(6)
#camera_style = vtkInteractorStyleTrackballActor()
#cam = renderer.GetActiveCamera()
#renderer.ResetCamera()
#cam.SetPosition(-215.8474015838563, -216.22066684736413, 445.22607604448297)
#cam.SetOrientation(-10.128997076024026, 45.60846498615863, 72.0405518553683)
#camera_style = vtkInteractorStyleTrackballActor()
#iren.SetInteractorStyle(camera_style)
#iren.AddObserver("LeftButtonPressEvent", OnPressLeftButton)

# Position is at origin, looking in z direction with y down
#print(cam.GetPosition())

iren.Initialize()

renWin.Render()
# screenshot code:
# w2if = vtkWindowToImageFilter()
# w2if.SetInput(renWin)
# w2if.SetInputBufferTypeToRGB()
# w2if.ReadFrontBufferOff()
# w2if.Update()

# writer = vtkPNGWriter()
# writer.SetFileName(subject+'_zoom_Screenshot.png')
# writer.SetInputConnection(w2if.GetOutputPort())
# writer.Write()
iren.Start()
