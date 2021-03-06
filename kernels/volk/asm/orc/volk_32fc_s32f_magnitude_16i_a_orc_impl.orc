.function volk_32fc_s32f_magnitude_16i_a_orc_impl
.source 8 src
.dest 2 dst
.floatparam 4 scalar
.temp 8 iqf
.temp 8 prodiqf
.temp 4 qf
.temp 4 if
.temp 4 sumf
.temp 4 rootf
.temp 4 rootl

x2 mulf prodiqf, src, src
splitql qf, if, prodiqf
addf sumf, if, qf
sqrtf rootf, sumf
mulf rootf, rootf, scalar
addf rootf, rootf, 0.5
convfl rootl, rootf
convsuslw dst, rootl
