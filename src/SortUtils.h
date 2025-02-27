namespace vkDelTet {
    
uint order_preserving_float_map(float value) {
    // For negative values, the mask becomes 0xffffffff.
    // For positive values, the mask becomes 0x80000000.
    uint uvalue = asuint(value);
    uint mask = -int(uvalue >> 31) | 0x80000000;
    return uvalue ^ mask;
}
float inverse_order_preserving_float_map(uint value) {
    // If the msb is set, the mask becomes 0x80000000.
    // If the msb is unset, the mask becomes 0xffffffff.
    uint mask = ((value >> 31) - 1) | 0x80000000;
    return asfloat(value ^ mask);
}

}