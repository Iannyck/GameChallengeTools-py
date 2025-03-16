import cv2 as cv
import numpy as np
import pyopencl as cl

from gamedifficulty.Constants import *
from gamedifficulty.Helpers import CoordsInImage

def DebugDisplay2ChannelImage(name, image):
    """
    Display a 2 channel image (float2) as a 3 channel image (float3) with the third channel being zero.
    :param image: the 2 channel image
    """
    image3 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    image3[:, :, :2] = image
    cv.imshow(name, image3)
    cv.waitKey(0)


class ProcessingContext:
    def __init__(self):
        self.clDevice = cl.get_platforms()[0].get_devices()[0]
        self.clContext = cl.Context([self.clDevice])

    def PropagateMovementPheromones(self, initialState: cv.Mat[cv.CV_8U], absorption: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:

        queue = cl.CommandQueue(self.clContext, self.clDevice)

        flip = False

        dt = 1.0/marioVelocity[0]

        program = cl.Program(self.clContext, """
        #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
        
        float DtForPixel(float2 source, int2 shape, int i, int j)
        {
            int index = i * shape.x + j;
            
            float2 vel = source[index];
            
            float dt = 1.0/vel.y;
            
            if (isnan(dt)) dt = 0.0;
            
            return dt;
        }
        
        __kernel void propagate(
            __global float2* source,
            __global float2* target,
            __global uchar* mask,
            __global int* changeCountBuffer,
            int2 shape,
            float gravity
            )
        {
            int i = get_global_id(1);
            int j = get_global_id(0);
        
            int index = i * shape.x + j;
            
            if (mask[index] > 0)
            {
                return;
            }
            
            // process :
            // - calculate dt using vertical axis (upwards) (the horizontal velocity is 0 so no divide)
            // - do the new velocity values, find the max for each axis
            // - check if a change is necessary :
            //  - if so increment atomic counter and change
            //  - else return
                  
            // right pixel
            target[index] = (float2)(i, j);
        }
        """).build()

        kernel = cl.Kernel(program, "propagate")

        print(initialState.shape)

        sourceImage = np.zeros(initialState.shape + (2, ), dtype=np.float32)
        sourceImage[initialState > 0] = (marioVelocity[1], marioVelocity[0]) # invert x and y

        DebugDisplay2ChannelImage("Source image", sourceImage)

        sourceBuffer = cl.Buffer(self.clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=sourceImage)
        targetBuffer = cl.Buffer(self.clContext, cl.mem_flags.READ_WRITE, size=sourceImage.nbytes)

        maskBuffer = cl.Buffer(self.clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=absorption)

        cv.imshow("absorption", absorption)
        cv.waitKey(0)

        changeCountBuffer = cl.Buffer(self.clContext, cl.mem_flags.READ_WRITE, size=4)

        # float2 are passed as numpy arrays
        imageShapeNp = np.array((initialState.shape[1], initialState.shape[0]), dtype=np.int32)
        gravityNp = np.float32(gravity)

        while True:

            kernel.set_arg(0, sourceBuffer)
            kernel.set_arg(1, targetBuffer)
            kernel.set_arg(2, maskBuffer)
            kernel.set_arg(3, changeCountBuffer)
            kernel.set_arg(4, imageShapeNp)
            kernel.set_arg(5, gravityNp)

            cl.enqueue_nd_range_kernel(queue, kernel, (initialState.shape[1], initialState.shape[0]), None)

            changeCount = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(queue, changeCount, changeCountBuffer, is_blocking=True)

            if changeCount[0] == 0:
                break

            flip = not flip

            if flip:
                kernel.set_arg(0, targetBuffer)
                kernel.set_arg(1, sourceBuffer)
            else:
                kernel.set_arg(0, sourceBuffer)
                kernel.set_arg(1, targetBuffer)

        # copy result back into float2
        cl.enqueue_copy(queue, sourceImage, sourceBuffer if flip else targetBuffer, is_blocking=True)

        DebugDisplay2ChannelImage("Result", sourceImage)

def CreateMaskFromPatternResult(detections: list[(int, int, int, int)], imageSize: (int, int)) -> cv.Mat[cv.CV_8U]:
    """
    Returns a mask from the detections positions. Returns 1 if the pixel is part of a detection, 0 otherwise.
    :param detections: the detections to create the mask from
    :param imageSize: the size of the image to create the mask from
    :return: the mask
    """
    result = np.zeros(imageSize, dtype=np.uint8)

    for (y, x, sizeY, sizeX) in detections:
        result[y:y + sizeY, x:x + sizeX] = 1

    return result

def CreatePlatformTextureFromMask(mask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Returns the top edges of the mask (the platforms the player can stand upon).
    1 if the pixel is the top edge of a platform, 0 otherwise.
    :param mask: a boolean texture, the mask to create the texture from
    :return: unsigned int 8 texture (grayscale)
    """
    kernel = np.array([[-1], [1]])

    return cv.filter2D(mask, -1, kernel)

def CreateJumpUpTexture() -> cv.Mat[cv.CV_8U]:
    result = np.zeros((int(jumpHeight), int(jumpHalfWidth * 2)), dtype=np.uint8)
    for y in range(result.shape[0]):
        hd = -(np.sqrt(2. * gravity * -y)) / gravity * marioVelocity[1]
        for x in range(int(jumpHalfWidth - hd)):
            result[y, x] = 1

    return result

def CreateJumpDownTexture() -> cv.Mat[cv.CV_8U]:
    result = np.zeros((int(jumpHeight), int(jumpHalfWidth * 2)), dtype=np.uint8)
    for y in range(result.shape[0]):
        hd = -(np.sqrt(2. * gravity * -y)) / gravity * marioVelocity[1]
        for x in range(int(jumpHalfWidth + hd)):
            result[y, x] = 1

    return result