#include <Rose/Core/DxgiFormatConvert.h>

using namespace RoseEngine;

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif
#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE uint16_t
#endif
#ifndef COMPRESS_FN
#define COMPRESS_FN (uint16_t)f32tof16(i)
#endif

ByteAddressBuffer inputData;
RWStructuredBuffer<OUTPUT_TYPE> outputData;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform uint count) {
    if (id.x < count) {
        const INPUT_TYPE i = inputData.Load<INPUT_TYPE>(id.x * sizeof(INPUT_TYPE));
        const OUTPUT_TYPE o = COMPRESS_FN;
        outputData[id.x] = o;
    }
}