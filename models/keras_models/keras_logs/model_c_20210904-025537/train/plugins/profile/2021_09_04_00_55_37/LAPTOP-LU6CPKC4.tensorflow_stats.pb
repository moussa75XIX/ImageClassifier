"?=
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1?O?????@A?O?????@a?&Ѵ???i?&Ѵ????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1?? ?r??@9?? ?r??@A?? ?r??@I?? ?r??@a
ģ?\??i
???V???Unknown
sHost_FusedMatMul"sequential_1/dense_2/Relu(1'1??@9'1??@A'1??@I'1??@aұ05a???i~???w???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1???K7Z?@9???K7Z?@A???K7Z?@I???K7Z?@a????)??iC??X???Unknown?
^HostGatherV2"GatherV2(1h??|?M?@9h??|?M?@Ah??|?M?@Ih??|?M?@a?nO?ks??i?N?z????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1+??Fv@9+??Fv@A+??Fv@I+??Fv@a?C??????i:?:?????Unknown
iHostWriteSummary"WriteSummary(1?Q??cS@9?Q??cS@A?Q??cS@I?Q??cS@ah???M?x?iir??}???Unknown?
s	HostDataset"Iterator::Model::ParallelMapV2(1?rh???J@9?rh???J@A?rh???J@I?rh???J@a?s????q?iP?5??(???Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333D@933333D@A?????IA@I?????IA@a????&f?iѷ	#????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1u?Vn@@9u?Vn@@Au?Vn@@Iu?Vn@@a ??1e?imՙK0T???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1?n??ZB@9?n??ZB@A{?G?z:@I{?G?z:@aBeX??`?ip:??&e???Unknown
qHostSoftmax"sequential_1/dense_3/Softmax(1sh??|?3@9sh??|?3@Ash??|?3@Ish??|?3@a3??!?Y?i?G?p?q???Unknown
vHost_FusedMatMul"sequential_1/dense_3/BiasAdd(1?Q???2@9?Q???2@A?Q???2@I?Q???2@a??+??W?i?)???}???Unknown
dHostDataset"Iterator::Model(1?MbX?P@9?MbX?P@Aףp=
?*@Iףp=
?*@aQ֓o?Q?i??\????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1fffffNW@9fffffNW@AB`??"(@IB`??"(@a????N?i???????Unknown
gHostStridedSlice"strided_slice(1w??/?'@9w??/?'@Aw??/?'@Iw??/?'@a?#??#?N?iT??q?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1h??|?u$@9h??|?u$@Ah??|?u$@Ih??|?u$@a<#??6J?i?5?F????Unknown
`HostGatherV2"
GatherV2_1(1??/?d"@9??/?d"@A??/?d"@I??/?d"@a??i	?G?i ??4+????Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1d;?O??!@9d;?O??!@Ad;?O??!@Id;?O??!@a4??A??F?i??߳ͧ???Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1
ףp=?@9
ףp=?@A
ףp=?@I
ףp=?@a??΂d4D?iB? ?ڬ???Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1?Q??k@9?Q??k@A?Q??k@I?Q??k@a????|C?i?????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1?A`??"@9?A`??"@A?A`??"@I?A`??"@aL?=&?B?i? ??d????Unknown
\HostArgMax"ArgMax_1(1Zd;??@9Zd;??@AZd;??@IZd;??@a?x?uB?i?^??????Unknown
ZHostArgMax"ArgMax(1?"??~j@9?"??~j@A?"??~j@I?"??~j@a?6?4B?i??3ʎ????Unknown
eHost
LogicalAnd"
LogicalAnd(1V-???@9V-???@AV-???@IV-???@aK?;[?A?i?;? ????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1J+?@9J+?@AJ+?@IJ+?@a?'?L@?ib??????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1?C?l?{@9?C?l?{@A?C?l?{@I?C?l?{@a?B???>?i
E/??????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_3/MatMul_1(1???(\?@9???(\?@A???(\?@I???(\?@ai??p?<?i?HHrl????Unknown
lHostIteratorGetNext"IteratorGetNext(1?z?G?@9?z?G?@A?z?G?@I?z?G?@a#X?ch<?iO?T?????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9??????@A??????@I??????@a??*+?;?i?Dk????Unknown
? HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1V-2@9V-2@AV-2@IV-2@a߂???8?iaRlG~????Unknown
?!HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?A`?ВD@9?A`?ВD@A??n??@I??n??@a?,??w?6?i'?bV????Unknown
V"HostSum"Sum_2(1?x?&1@9?x?&1@A?x?&1@I?x?&1@aU???P?5?i?y`????Unknown
V#HostCast"Cast(1y?&1?@9y?&1?@Ay?&1?@Iy?&1?@a"B52=\5?i?b??????Unknown
?$HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?A`??"@9?A`??"@A?A`??"@I?A`??"@axW??$N3?i???%????Unknown
?%HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@a?f?&3?i?+݅?????Unknown
?&HostReluGrad"+gradient_tape/sequential_1/dense_2/ReluGrad(1!?rh??@9!?rh??@A!?rh??@I!?rh??@a>X|$`-?i?C?`????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_1(1?Q???@9?Q???@A?Q???@I?Q???@a??H?]-?i?a`6????Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1???Q?@9???Q?@A???Q?@I???Q?@a?ï???+?i?@??????Unknown
o)HostCast"categorical_crossentropy/Cast(1?E????@9?E????@A?E????@I?E????@aZ5? (?i~8d?u????Unknown
X*HostEqual"Equal(1?ʡE??@9?ʡE??@A?ʡE??@I?ʡE??@ai?? '?iZp??????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_4(11?Zd@91?Zd@A1?Zd@I1?Zd@a??%3dH&?iz??1J????Unknown
X,HostCast"Cast_2(1m???????9m???????Am???????Im???????a@?ZL?#?i(RX?{????Unknown
?-HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1??|?5^??9??|?5^??A??|?5^??I??|?5^??a6	??3,"?iiz???????Unknown
t.HostAssignAddVariableOp"AssignAddVariableOp(1sh??|???9sh??|???Ash??|???Ish??|???aX??t!?ii???????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_3(1J+???9J+???AJ+???IJ+???a???IZ!?i?\???????Unknown
?0HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1!?rh????9!?rh????A!?rh????I!?rh????a?l??@!?i?ug??????Unknown
?1HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1???S???9???S???A???S???I???S???at?g???i??|j?????Unknown
X2HostCast"Cast_3(1??n????9??n????A??n????I??n????atN?j&?i??͝?????Unknown
s3HostReadVariableOp"SGD/Cast/ReadVariableOp(1㥛? ???9㥛? ???A㥛? ???I㥛? ???a?8??q??i??[??????Unknown
u4HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1?ʡE????9?ʡE????A?ʡE????I?ʡE????ai?? ?i???I????Unknown
{5HostSum"*categorical_crossentropy/weighted_loss/Sum(1?n?????9?n?????A?n?????I?n?????a?QP???i??? ????Unknown
y6HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a??BjS??i??/Y?????Unknown
?7HostReadVariableOp"*sequential_1/dense_2/MatMul/ReadVariableOp(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a??BjS??i?1˫H????Unknown
a8HostIdentity"Identity(1X9??v???9X9??v???AX9??v???IX9??v???au[F??U?i??Z?????Unknown?
`9HostDivNoNan"
div_no_nan(1Zd;?O???9Zd;?O???AZd;?O???IZd;?O???a@[??a??iD٠?x????Unknown
?:HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1?$??C??9?$??C??A?$??C??I?$??C??a?;v?&w?i???F????Unknown
b;HostDivNoNan"div_no_nan_1(1D?l?????9D?l?????AD?l?????ID?l?????aLU???p?i1??̇????Unknown
T<HostMul"Mul(1o??ʡ??9o??ʡ??Ao??ʡ??Io??ʡ??a???k?iW)????Unknown
?=HostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1??ʡE???9??ʡE???A??ʡE???I??ʡE???a?ɞ?Ha?i?}&??????Unknown
w>HostReadVariableOp"div_no_nan_1/ReadVariableOp(1;?O??n??9;?O??n??A;?O??n??I;?O??n??ab%?9u??ild???????Unknown
w?HostReadVariableOp"div_no_nan/ReadVariableOp_1(1P??n???9P??n???AP??n???IP??n???a?.???G
?i????`????Unknown
u@HostReadVariableOp"div_no_nan/ReadVariableOp(1????x???9????x???A????x???I????x???a?an?:?i????????Unknown
?AHostDivNoNan",categorical_crossentropy/weighted_loss/value(1R???Q??9R???Q??AR???Q??IR???Q??aL1???(?>i?????????Unknown*?<
}HostMatMul")gradient_tape/sequential_1/dense_2/MatMul(1?? ?r??@9?? ?r??@A?? ?r??@I?? ?r??@a>???	??i>???	???Unknown
sHost_FusedMatMul"sequential_1/dense_2/Relu(1'1??@9'1??@A'1??@I'1??@a????j??i-A??L????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1???K7Z?@9???K7Z?@A???K7Z?@I???K7Z?@aMVy????i???ƽ???Unknown?
^HostGatherV2"GatherV2(1h??|?M?@9h??|?M?@Ah??|?M?@Ih??|?M?@a7??&????iN??(????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1+??Fv@9+??Fv@A+??Fv@I+??Fv@aSr$??Ӧ?isC0?f!???Unknown
iHostWriteSummary"WriteSummary(1?Q??cS@9?Q??cS@A?Q??cS@I?Q??cS@au??Mރ?i????p???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1?rh???J@9?rh???J@A?rh???J@I?rh???J@a ?1?p?{?i?J?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(133333D@933333D@A?????IA@I?????IA@a?9???q?iF??}????Unknown
?	HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1u?Vn@@9u?Vn@@Au?Vn@@Iu?Vn@@aC#?c?p?i???)????Unknown
?
HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1?n??ZB@9?n??ZB@A{?G?z:@I{?G?z:@aS?8?"k?i?sDgL???Unknown
qHostSoftmax"sequential_1/dense_3/Softmax(1sh??|?3@9sh??|?3@Ash??|?3@Ish??|?3@a?m?[!~d?i???????Unknown
vHost_FusedMatMul"sequential_1/dense_3/BiasAdd(1?Q???2@9?Q???2@A?Q???2@I?Q???2@a?A?? c?i :?~?/???Unknown
dHostDataset"Iterator::Model(1?MbX?P@9?MbX?P@Aףp=
?*@Iףp=
?*@a{J?m?[?iE?Y5k=???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1fffffNW@9fffffNW@AB`??"(@IB`??"(@a?BU˳X?i?e??I???Unknown
gHostStridedSlice"strided_slice(1w??/?'@9w??/?'@Aw??/?'@Iw??/?'@a
?~OtX?iD%fB?U???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1h??|?u$@9h??|?u$@Ah??|?u$@Ih??|?u$@a?v???T?iL?:?z`???Unknown
`HostGatherV2"
GatherV2_1(1??/?d"@9??/?d"@A??/?d"@I??/?d"@aSO*<Y?R?it??U?i???Unknown
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1d;?O??!@9d;?O??!@Ad;?O??!@Id;?O??!@a??|,?R?i?3o??r???Unknown
|HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1
ףp=?@9
ףp=?@A
ףp=?@I
ףp=?@a:???(P?i???N?z???Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1?Q??k@9?Q??k@A?Q??k@I?Q??k@a??`?),O?i?O5Yʂ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1?A`??"@9?A`??"@A?A`??"@I?A`??"@a????S?M?i0".A????Unknown
\HostArgMax"ArgMax_1(1Zd;??@9Zd;??@AZd;??@IZd;??@a??T?M?iK?Sâ????Unknown
ZHostArgMax"ArgMax(1?"??~j@9?"??~j@A?"??~j@I?"??~j@a?Al?rM?i[??_?????Unknown
eHost
LogicalAnd"
LogicalAnd(1V-???@9V-???@AV-???@IV-???@a?????L?iW???????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1J+?@9J+?@AJ+?@IJ+?@a*?lg?I?iS???????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1?C?l?{@9?C?l?{@A?C?l?{@I?C?l?{@a?ѬL?H?i??h?????Unknown
HostMatMul"+gradient_tape/sequential_1/dense_3/MatMul_1(1???(\?@9???(\?@A???(\?@I???(\?@a???:G?i;???K????Unknown
lHostIteratorGetNext"IteratorGetNext(1?z?G?@9?z?G?@A?z?G?@I?z?G?@aײQ-?kF?i?3???????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9??????@A??????@I??????@a??2??VF?ig?^>|????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_1/dense_2/BiasAdd/BiasAddGrad(1V-2@9V-2@AV-2@IV-2@a???K??C?i_??-g????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?A`?ВD@9?A`?ВD@A??n??@I??n??@a80?R?0B?i?dFi?????Unknown
V HostSum"Sum_2(1?x?&1@9?x?&1@A?x?&1@I?x?&1@a5??&tA?iY?lP????Unknown
V!HostCast"Cast(1y?&1?@9y?&1?@Ay?&1?@Iy?&1?@a?z???A?i85?ӕ????Unknown
?"HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?A`??"@9?A`??"@A?A`??"@I?A`??"@a?????>?iO??r????Unknown
?#HostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@a??$)??>?i??5_F????Unknown
?$HostReluGrad"+gradient_tape/sequential_1/dense_2/ReluGrad(1!?rh??@9!?rh??@A!?rh??@I!?rh??@a????~7?i2??76????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_1(1?Q???@9?Q???@A?Q???@I?Q???@aOᥪ|7?iT_?%????Unknown
?&HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1???Q?@9???Q?@A???Q?@I???Q?@a?B?A6?i?A??????Unknown
o'HostCast"categorical_crossentropy/Cast(1?E????@9?E????@A?E????@I?E????@a???@L3?i??ʐW????Unknown
X(HostEqual"Equal(1?ʡE??@9?ʡE??@A?ʡE??@I?ʡE??@a????ee2?i?-?=?????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_4(11?Zd@91?Zd@A1?Zd@I1?Zd@a????}?1?i??:??????Unknown
X*HostCast"Cast_2(1m???????9m???????Am???????Im???????apP??Ϗ.?i?B8??????Unknown
?+HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1??|?5^??9??|?5^??A??|?5^??I??|?5^??a?l???-?i觘????Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1sh??|???9sh??|???Ash??|???Ish??|???a5[E?
?+?iY?hW????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_3(1J+???9J+???AJ+???IJ+???a????+?iئ??????Unknown
?.HostReadVariableOp"+sequential_1/dense_2/BiasAdd/ReadVariableOp(1!?rh????9!?rh????A!?rh????I!?rh????a?zʣ?+?i??=?????Unknown
?/HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1???S???9???S???A???S???I???S???ay&;͚z(?i2???T????Unknown
X0HostCast"Cast_3(1??n????9??n????A??n????I??n????aE#?=?P'?iԒn??????Unknown
s1HostReadVariableOp"SGD/Cast/ReadVariableOp(1㥛? ???9㥛? ???A㥛? ???I㥛? ???a?an?#3%?i????????Unknown
u2HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1?ʡE????9?ʡE????A?ʡE????I?ʡE????a????ee"?i*?GC????Unknown
{3HostSum"*categorical_crossentropy/weighted_loss/Sum(1?n?????9?n?????A?n?????I?n?????aH?I?l;"?iÆ??f????Unknown
y4HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a̹???m ?iQ?m????Unknown
?5HostReadVariableOp"*sequential_1/dense_2/MatMul/ReadVariableOp(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a̹???m ?i{{дt????Unknown
a6HostIdentity"Identity(1X9??v???9X9??v???AX9??v???IX9??v???a?IhѾC ?i ???x????Unknown?
`7HostDivNoNan"
div_no_nan(1Zd;?O???9Zd;?O???AZd;?O???IZd;?O???a	J???;?ijY?Z????Unknown
?8HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1?$??C??9?$??C??A?$??C??I?$??C??a?Lt=??i,??P:????Unknown
b9HostDivNoNan"div_no_nan_1(1D?l?????9D?l?????AD?l?????ID?l?????a?????L?i???????Unknown
T:HostMul"Mul(1o??ʡ??9o??ʡ??Ao??ʡ??Io??ʡ??aU?D?i???????Unknown
?;HostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1??ʡE???9??ʡE???A??ʡE???I??ʡE???a??*#oL?iq1o9?????Unknown
w<HostReadVariableOp"div_no_nan_1/ReadVariableOp(1;?O??n??9;?O??n??A;?O??n??I;?O??n??a?B?????ik??Y????Unknown
w=HostReadVariableOp"div_no_nan/ReadVariableOp_1(1P??n???9P??n???AP??n???IP??n???a ^&??i[?_F????Unknown
u>HostReadVariableOp"div_no_nan/ReadVariableOp(1????x???9????x???A????x???I????x???a۝??<a?i.EP?????Unknown
??HostDivNoNan",categorical_crossentropy/weighted_loss/value(1R???Q??9R???Q??AR???Q??IR???Q??aH?z????i     ???Unknown2Nvidia GPU (Turing)