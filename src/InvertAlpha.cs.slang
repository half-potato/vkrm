RWTexture2D<float4> image;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId: SV_DispatchThreadID, uniform uint2 dim) {
    if (any(threadId.xy >= dim))
        return;
    float4 c = image[threadId.xy];
    image[threadId.xy] = float4(c.rgb, 1 - c.a);
}