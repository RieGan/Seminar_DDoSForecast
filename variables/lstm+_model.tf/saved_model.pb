??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *'
shared_nameembedding_2/embeddings
?
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes
:	? *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
?
rnn/peephole_lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*.
shared_namernn/peephole_lstm_cell/kernel
?
1rnn/peephole_lstm_cell/kernel/Read/ReadVariableOpReadVariableOprnn/peephole_lstm_cell/kernel*
_output_shapes
:	 ?*
dtype0
?
'rnn/peephole_lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*8
shared_name)'rnn/peephole_lstm_cell/recurrent_kernel
?
;rnn/peephole_lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp'rnn/peephole_lstm_cell/recurrent_kernel*
_output_shapes
:	 ?*
dtype0
?
rnn/peephole_lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namernn/peephole_lstm_cell/bias
?
/rnn/peephole_lstm_cell/bias/Read/ReadVariableOpReadVariableOprnn/peephole_lstm_cell/bias*
_output_shapes	
:?*
dtype0
?
2rnn/peephole_lstm_cell/input_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42rnn/peephole_lstm_cell/input_gate_peephole_weights
?
Frnn/peephole_lstm_cell/input_gate_peephole_weights/Read/ReadVariableOpReadVariableOp2rnn/peephole_lstm_cell/input_gate_peephole_weights*
_output_shapes
: *
dtype0
?
3rnn/peephole_lstm_cell/forget_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53rnn/peephole_lstm_cell/forget_gate_peephole_weights
?
Grnn/peephole_lstm_cell/forget_gate_peephole_weights/Read/ReadVariableOpReadVariableOp3rnn/peephole_lstm_cell/forget_gate_peephole_weights*
_output_shapes
: *
dtype0
?
3rnn/peephole_lstm_cell/output_gate_peephole_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53rnn/peephole_lstm_cell/output_gate_peephole_weights
?
Grnn/peephole_lstm_cell/output_gate_peephole_weights/Read/ReadVariableOpReadVariableOp3rnn/peephole_lstm_cell/output_gate_peephole_weights*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
#SGD/embedding_2/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *4
shared_name%#SGD/embedding_2/embeddings/momentum
?
7SGD/embedding_2/embeddings/momentum/Read/ReadVariableOpReadVariableOp#SGD/embedding_2/embeddings/momentum*
_output_shapes
:	? *
dtype0
?
SGD/dense_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *,
shared_nameSGD/dense_3/kernel/momentum
?
/SGD/dense_3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/kernel/momentum*
_output_shapes

: *
dtype0
?
SGD/dense_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_3/bias/momentum
?
-SGD/dense_3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/bias/momentum*
_output_shapes
:*
dtype0
?
*SGD/rnn/peephole_lstm_cell/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*;
shared_name,*SGD/rnn/peephole_lstm_cell/kernel/momentum
?
>SGD/rnn/peephole_lstm_cell/kernel/momentum/Read/ReadVariableOpReadVariableOp*SGD/rnn/peephole_lstm_cell/kernel/momentum*
_output_shapes
:	 ?*
dtype0
?
4SGD/rnn/peephole_lstm_cell/recurrent_kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*E
shared_name64SGD/rnn/peephole_lstm_cell/recurrent_kernel/momentum
?
HSGD/rnn/peephole_lstm_cell/recurrent_kernel/momentum/Read/ReadVariableOpReadVariableOp4SGD/rnn/peephole_lstm_cell/recurrent_kernel/momentum*
_output_shapes
:	 ?*
dtype0
?
(SGD/rnn/peephole_lstm_cell/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*9
shared_name*(SGD/rnn/peephole_lstm_cell/bias/momentum
?
<SGD/rnn/peephole_lstm_cell/bias/momentum/Read/ReadVariableOpReadVariableOp(SGD/rnn/peephole_lstm_cell/bias/momentum*
_output_shapes	
:?*
dtype0
?
?SGD/rnn/peephole_lstm_cell/input_gate_peephole_weights/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?SGD/rnn/peephole_lstm_cell/input_gate_peephole_weights/momentum
?
SSGD/rnn/peephole_lstm_cell/input_gate_peephole_weights/momentum/Read/ReadVariableOpReadVariableOp?SGD/rnn/peephole_lstm_cell/input_gate_peephole_weights/momentum*
_output_shapes
: *
dtype0
?
@SGD/rnn/peephole_lstm_cell/forget_gate_peephole_weights/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@SGD/rnn/peephole_lstm_cell/forget_gate_peephole_weights/momentum
?
TSGD/rnn/peephole_lstm_cell/forget_gate_peephole_weights/momentum/Read/ReadVariableOpReadVariableOp@SGD/rnn/peephole_lstm_cell/forget_gate_peephole_weights/momentum*
_output_shapes
: *
dtype0
?
@SGD/rnn/peephole_lstm_cell/output_gate_peephole_weights/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@SGD/rnn/peephole_lstm_cell/output_gate_peephole_weights/momentum
?
TSGD/rnn/peephole_lstm_cell/output_gate_peephole_weights/momentum/Read/ReadVariableOpReadVariableOp@SGD/rnn/peephole_lstm_cell/output_gate_peephole_weights/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
?+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?*
value?*B?* B?*
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
b

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?
 iter
	!decay
"learning_rate
#momentummomentumRmomentumSmomentumT$momentumU%momentumV&momentumW'momentumX(momentumY)momentumZ
 
?
0
$1
%2
&3
'4
(5
)6
7
8
?
0
$1
%2
&3
'4
(5
)6
7
8
?
*layer_metrics

+layers
regularization_losses
,non_trainable_variables
	variables
-layer_regularization_losses
trainable_variables
.metrics
 
fd
VARIABLE_VALUEembedding_2/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
/layer_metrics

0layers
regularization_losses
1non_trainable_variables
	variables
2layer_regularization_losses
trainable_variables
3metrics
 
 
 
?
4layer_metrics

5layers
regularization_losses
6non_trainable_variables
	variables
7layer_regularization_losses
trainable_variables
8metrics
?

$kernel
%recurrent_kernel
&bias
'input_gate_peephole_weights
 (forget_gate_peephole_weights
 )output_gate_peephole_weights
9regularization_losses
:	variables
;trainable_variables
<	keras_api
 
 
*
$0
%1
&2
'3
(4
)5
*
$0
%1
&2
'3
(4
)5
?
=layer_metrics

>states

?layers
regularization_losses
@non_trainable_variables
	variables
Alayer_regularization_losses
trainable_variables
Bmetrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Clayer_metrics

Dlayers
regularization_losses
Enon_trainable_variables
	variables
Flayer_regularization_losses
trainable_variables
Gmetrics
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErnn/peephole_lstm_cell/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'rnn/peephole_lstm_cell/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUErnn/peephole_lstm_cell/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2rnn/peephole_lstm_cell/input_gate_peephole_weights&variables/4/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3rnn/peephole_lstm_cell/forget_gate_peephole_weights&variables/5/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3rnn/peephole_lstm_cell/output_gate_peephole_weights&variables/6/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3
 
 

H0
 
 
 
 
 
 
 
 
 
 
 
*
$0
%1
&2
'3
(4
)5
*
$0
%1
&2
'3
(4
)5
?
Ilayer_metrics

Jlayers
9regularization_losses
Knon_trainable_variables
:	variables
Llayer_regularization_losses
;trainable_variables
Mmetrics
 
 

0
 
 
 
 
 
 
 
 
4
	Ntotal
	Ocount
P	variables
Q	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

P	variables
??
VARIABLE_VALUE#SGD/embedding_2/embeddings/momentum]layer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_3/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_3/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*SGD/rnn/peephole_lstm_cell/kernel/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4SGD/rnn/peephole_lstm_cell/recurrent_kernel/momentumIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE(SGD/rnn/peephole_lstm_cell/bias/momentumIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?SGD/rnn/peephole_lstm_cell/input_gate_peephole_weights/momentumIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@SGD/rnn/peephole_lstm_cell/forget_gate_peephole_weights/momentumIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@SGD/rnn/peephole_lstm_cell/output_gate_peephole_weights/momentumIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
?
!serving_default_embedding_2_inputPlaceholder*'
_output_shapes
:?????????2*
dtype0*
shape:?????????2
?
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_2_inputembedding_2/embeddingsrnn/peephole_lstm_cell/kernel'rnn/peephole_lstm_cell/recurrent_kernelrnn/peephole_lstm_cell/bias2rnn/peephole_lstm_cell/input_gate_peephole_weights3rnn/peephole_lstm_cell/forget_gate_peephole_weights3rnn/peephole_lstm_cell/output_gate_peephole_weightsdense_3/kerneldense_3/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_29210
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_2/embeddings/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp1rnn/peephole_lstm_cell/kernel/Read/ReadVariableOp;rnn/peephole_lstm_cell/recurrent_kernel/Read/ReadVariableOp/rnn/peephole_lstm_cell/bias/Read/ReadVariableOpFrnn/peephole_lstm_cell/input_gate_peephole_weights/Read/ReadVariableOpGrnn/peephole_lstm_cell/forget_gate_peephole_weights/Read/ReadVariableOpGrnn/peephole_lstm_cell/output_gate_peephole_weights/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7SGD/embedding_2/embeddings/momentum/Read/ReadVariableOp/SGD/dense_3/kernel/momentum/Read/ReadVariableOp-SGD/dense_3/bias/momentum/Read/ReadVariableOp>SGD/rnn/peephole_lstm_cell/kernel/momentum/Read/ReadVariableOpHSGD/rnn/peephole_lstm_cell/recurrent_kernel/momentum/Read/ReadVariableOp<SGD/rnn/peephole_lstm_cell/bias/momentum/Read/ReadVariableOpSSGD/rnn/peephole_lstm_cell/input_gate_peephole_weights/momentum/Read/ReadVariableOpTSGD/rnn/peephole_lstm_cell/forget_gate_peephole_weights/momentum/Read/ReadVariableOpTSGD/rnn/peephole_lstm_cell/output_gate_peephole_weights/momentum/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_30803
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_2/embeddingsdense_3/kerneldense_3/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumrnn/peephole_lstm_cell/kernel'rnn/peephole_lstm_cell/recurrent_kernelrnn/peephole_lstm_cell/bias2rnn/peephole_lstm_cell/input_gate_peephole_weights3rnn/peephole_lstm_cell/forget_gate_peephole_weights3rnn/peephole_lstm_cell/output_gate_peephole_weightstotalcount#SGD/embedding_2/embeddings/momentumSGD/dense_3/kernel/momentumSGD/dense_3/bias/momentum*SGD/rnn/peephole_lstm_cell/kernel/momentum4SGD/rnn/peephole_lstm_cell/recurrent_kernel/momentum(SGD/rnn/peephole_lstm_cell/bias/momentum?SGD/rnn/peephole_lstm_cell/input_gate_peephole_weights/momentum@SGD/rnn/peephole_lstm_cell/forget_gate_peephole_weights/momentum@SGD/rnn/peephole_lstm_cell/output_gate_peephole_weights/momentum*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_30885??
?d
?
while_body_28694
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_peephole_lstm_cell_matmul_readvariableop_resource_0?
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0>
:while_peephole_lstm_cell_biasadd_readvariableop_resource_06
2while_peephole_lstm_cell_readvariableop_resource_08
4while_peephole_lstm_cell_readvariableop_1_resource_08
4while_peephole_lstm_cell_readvariableop_2_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_peephole_lstm_cell_matmul_readvariableop_resource=
9while_peephole_lstm_cell_matmul_1_readvariableop_resource<
8while_peephole_lstm_cell_biasadd_readvariableop_resource4
0while_peephole_lstm_cell_readvariableop_resource6
2while_peephole_lstm_cell_readvariableop_1_resource6
2while_peephole_lstm_cell_readvariableop_2_resource??/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?.while/peephole_lstm_cell/MatMul/ReadVariableOp?0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?'while/peephole_lstm_cell/ReadVariableOp?)while/peephole_lstm_cell/ReadVariableOp_1?)while/peephole_lstm_cell/ReadVariableOp_2?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOp?
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/peephole_lstm_cell/MatMul?
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/peephole_lstm_cell/MatMul_1?
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/peephole_lstm_cell/add?
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/peephole_lstm_cell/BiasAdd?
while/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
while/peephole_lstm_cell/Const?
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim?
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2 
while/peephole_lstm_cell/split?
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp?
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/mul?
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_1?
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 while/peephole_lstm_cell/Sigmoid?
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1?
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_1?
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_2?
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_1?
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_2?
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/Tanh?
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_3?
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_3?
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2?
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_4?
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_4?
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_2?
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2!
while/peephole_lstm_cell/Tanh_1?
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?	
?
while_cond_30233
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_30233___redundant_placeholder03
/while_while_cond_30233___redundant_placeholder13
/while_while_cond_30233___redundant_placeholder23
/while_while_cond_30233___redundant_placeholder33
/while_while_cond_30233___redundant_placeholder43
/while_while_cond_30233___redundant_placeholder53
/while_while_cond_30233___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
m
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_28598

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????2 2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2 :S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?d
?
while_body_30416
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_peephole_lstm_cell_matmul_readvariableop_resource_0?
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0>
:while_peephole_lstm_cell_biasadd_readvariableop_resource_06
2while_peephole_lstm_cell_readvariableop_resource_08
4while_peephole_lstm_cell_readvariableop_1_resource_08
4while_peephole_lstm_cell_readvariableop_2_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_peephole_lstm_cell_matmul_readvariableop_resource=
9while_peephole_lstm_cell_matmul_1_readvariableop_resource<
8while_peephole_lstm_cell_biasadd_readvariableop_resource4
0while_peephole_lstm_cell_readvariableop_resource6
2while_peephole_lstm_cell_readvariableop_1_resource6
2while_peephole_lstm_cell_readvariableop_2_resource??/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?.while/peephole_lstm_cell/MatMul/ReadVariableOp?0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?'while/peephole_lstm_cell/ReadVariableOp?)while/peephole_lstm_cell/ReadVariableOp_1?)while/peephole_lstm_cell/ReadVariableOp_2?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOp?
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/peephole_lstm_cell/MatMul?
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/peephole_lstm_cell/MatMul_1?
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/peephole_lstm_cell/add?
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/peephole_lstm_cell/BiasAdd?
while/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
while/peephole_lstm_cell/Const?
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim?
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2 
while/peephole_lstm_cell/split?
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp?
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/mul?
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_1?
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 while/peephole_lstm_cell/Sigmoid?
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1?
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_1?
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_2?
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_1?
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_2?
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/Tanh?
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_3?
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_3?
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2?
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_4?
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_4?
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_2?
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2!
while/peephole_lstm_cell/Tanh_1?
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?	
?
F__inference_embedding_2_layer_call_and_return_conditional_losses_28565

inputs
embedding_lookup_28559
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????22
Cast?
embedding_lookupResourceGatherembedding_lookup_28559Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/28559*+
_output_shapes
:?????????2 *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/28559*+
_output_shapes
:?????????2 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2 2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29080
embedding_2_input
embedding_2_29057
	rnn_29061
	rnn_29063
	rnn_29065
	rnn_29067
	rnn_29069
	rnn_29071
dense_3_29074
dense_3_29076
identity??dense_3/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?rnn/StatefulPartitionedCall?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallembedding_2_inputembedding_2_29057*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_285652%
#embedding_2/StatefulPartitionedCall?
#spatial_dropout1d_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_286032%
#spatial_dropout1d_2/PartitionedCall?
rnn/StatefulPartitionedCallStatefulPartitionedCall,spatial_dropout1d_2/PartitionedCall:output:0	rnn_29061	rnn_29063	rnn_29065	rnn_29067	rnn_29069	rnn_29071*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_289782
rnn/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_3_29074dense_3_29076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_290372!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:Z V
'
_output_shapes
:?????????2
+
_user_specified_nameembedding_2_input
??
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29423

inputs&
"embedding_2_embedding_lookup_292149
5rnn_peephole_lstm_cell_matmul_readvariableop_resource;
7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource:
6rnn_peephole_lstm_cell_biasadd_readvariableop_resource2
.rnn_peephole_lstm_cell_readvariableop_resource4
0rnn_peephole_lstm_cell_readvariableop_1_resource4
0rnn_peephole_lstm_cell_readvariableop_2_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_2/embedding_lookup?-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp?,rnn/peephole_lstm_cell/MatMul/ReadVariableOp?.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp?%rnn/peephole_lstm_cell/ReadVariableOp?'rnn/peephole_lstm_cell/ReadVariableOp_1?'rnn/peephole_lstm_cell/ReadVariableOp_2?	rnn/whileu
embedding_2/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????22
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_29214embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_2/embedding_lookup/29214*+
_output_shapes
:?????????2 *
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/29214*+
_output_shapes
:?????????2 2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2 2)
'embedding_2/embedding_lookup/Identity_1?
spatial_dropout1d_2/ShapeShape0embedding_2/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
spatial_dropout1d_2/Shape?
'spatial_dropout1d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'spatial_dropout1d_2/strided_slice/stack?
)spatial_dropout1d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout1d_2/strided_slice/stack_1?
)spatial_dropout1d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout1d_2/strided_slice/stack_2?
!spatial_dropout1d_2/strided_sliceStridedSlice"spatial_dropout1d_2/Shape:output:00spatial_dropout1d_2/strided_slice/stack:output:02spatial_dropout1d_2/strided_slice/stack_1:output:02spatial_dropout1d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!spatial_dropout1d_2/strided_slice?
)spatial_dropout1d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout1d_2/strided_slice_1/stack?
+spatial_dropout1d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+spatial_dropout1d_2/strided_slice_1/stack_1?
+spatial_dropout1d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+spatial_dropout1d_2/strided_slice_1/stack_2?
#spatial_dropout1d_2/strided_slice_1StridedSlice"spatial_dropout1d_2/Shape:output:02spatial_dropout1d_2/strided_slice_1/stack:output:04spatial_dropout1d_2/strided_slice_1/stack_1:output:04spatial_dropout1d_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#spatial_dropout1d_2/strided_slice_1?
!spatial_dropout1d_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!spatial_dropout1d_2/dropout/Const?
spatial_dropout1d_2/dropout/MulMul0embedding_2/embedding_lookup/Identity_1:output:0*spatial_dropout1d_2/dropout/Const:output:0*
T0*+
_output_shapes
:?????????2 2!
spatial_dropout1d_2/dropout/Mul?
2spatial_dropout1d_2/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2spatial_dropout1d_2/dropout/random_uniform/shape/1?
0spatial_dropout1d_2/dropout/random_uniform/shapePack*spatial_dropout1d_2/strided_slice:output:0;spatial_dropout1d_2/dropout/random_uniform/shape/1:output:0,spatial_dropout1d_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
:22
0spatial_dropout1d_2/dropout/random_uniform/shape?
8spatial_dropout1d_2/dropout/random_uniform/RandomUniformRandomUniform9spatial_dropout1d_2/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02:
8spatial_dropout1d_2/dropout/random_uniform/RandomUniform?
*spatial_dropout1d_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2,
*spatial_dropout1d_2/dropout/GreaterEqual/y?
(spatial_dropout1d_2/dropout/GreaterEqualGreaterEqualAspatial_dropout1d_2/dropout/random_uniform/RandomUniform:output:03spatial_dropout1d_2/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2*
(spatial_dropout1d_2/dropout/GreaterEqual?
 spatial_dropout1d_2/dropout/CastCast,spatial_dropout1d_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2"
 spatial_dropout1d_2/dropout/Cast?
!spatial_dropout1d_2/dropout/Mul_1Mul#spatial_dropout1d_2/dropout/Mul:z:0$spatial_dropout1d_2/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2 2#
!spatial_dropout1d_2/dropout/Mul_1k
	rnn/ShapeShape%spatial_dropout1d_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack?
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1?
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2?
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/packed/1?
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const?
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
	rnn/zerosh
rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros_1/mul/y?
rnn/zeros_1/mulMulrnn/strided_slice:output:0rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/mulk
rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
rnn/zeros_1/Less/y
rnn/zeros_1/LessLessrnn/zeros_1/mul:z:0rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/Lessn
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros_1/packed/1?
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_1/packedk
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_1/Const?
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
rnn/zeros_1}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm?
rnn/transpose	Transpose%spatial_dropout1d_2/dropout/Mul_1:z:0rnn/transpose/perm:output:0*
T0*+
_output_shapes
:2????????? 2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1?
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack?
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1?
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2?
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1?
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
rnn/TensorArrayV2/element_shape?
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2?
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor?
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack?
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1?
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
rnn/strided_slice_2?
,rnn/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp5rnn_peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02.
,rnn/peephole_lstm_cell/MatMul/ReadVariableOp?
rnn/peephole_lstm_cell/MatMulMatMulrnn/strided_slice_2:output:04rnn/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
rnn/peephole_lstm_cell/MatMul?
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype020
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp?
rnn/peephole_lstm_cell/MatMul_1MatMulrnn/zeros:output:06rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
rnn/peephole_lstm_cell/MatMul_1?
rnn/peephole_lstm_cell/addAddV2'rnn/peephole_lstm_cell/MatMul:product:0)rnn/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
rnn/peephole_lstm_cell/add?
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6rnn_peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp?
rnn/peephole_lstm_cell/BiasAddBiasAddrnn/peephole_lstm_cell/add:z:05rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
rnn/peephole_lstm_cell/BiasAdd~
rnn/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/peephole_lstm_cell/Const?
&rnn/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&rnn/peephole_lstm_cell/split/split_dim?
rnn/peephole_lstm_cell/splitSplit/rnn/peephole_lstm_cell/split/split_dim:output:0'rnn/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
rnn/peephole_lstm_cell/split?
%rnn/peephole_lstm_cell/ReadVariableOpReadVariableOp.rnn_peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02'
%rnn/peephole_lstm_cell/ReadVariableOp?
rnn/peephole_lstm_cell/mulMul-rnn/peephole_lstm_cell/ReadVariableOp:value:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul?
rnn/peephole_lstm_cell/add_1AddV2%rnn/peephole_lstm_cell/split:output:0rnn/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/add_1?
rnn/peephole_lstm_cell/SigmoidSigmoid rnn/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2 
rnn/peephole_lstm_cell/Sigmoid?
'rnn/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp0rnn_peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'rnn/peephole_lstm_cell/ReadVariableOp_1?
rnn/peephole_lstm_cell/mul_1Mul/rnn/peephole_lstm_cell/ReadVariableOp_1:value:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_1?
rnn/peephole_lstm_cell/add_2AddV2%rnn/peephole_lstm_cell/split:output:1 rnn/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/add_2?
 rnn/peephole_lstm_cell/Sigmoid_1Sigmoid rnn/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2"
 rnn/peephole_lstm_cell/Sigmoid_1?
rnn/peephole_lstm_cell/mul_2Mul$rnn/peephole_lstm_cell/Sigmoid_1:y:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_2?
rnn/peephole_lstm_cell/TanhTanh%rnn/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/Tanh?
rnn/peephole_lstm_cell/mul_3Mul"rnn/peephole_lstm_cell/Sigmoid:y:0rnn/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_3?
rnn/peephole_lstm_cell/add_3AddV2 rnn/peephole_lstm_cell/mul_2:z:0 rnn/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/add_3?
'rnn/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp0rnn_peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02)
'rnn/peephole_lstm_cell/ReadVariableOp_2?
rnn/peephole_lstm_cell/mul_4Mul/rnn/peephole_lstm_cell/ReadVariableOp_2:value:0 rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_4?
rnn/peephole_lstm_cell/add_4AddV2%rnn/peephole_lstm_cell/split:output:3 rnn/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/add_4?
 rnn/peephole_lstm_cell/Sigmoid_2Sigmoid rnn/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2"
 rnn/peephole_lstm_cell/Sigmoid_2?
rnn/peephole_lstm_cell/Tanh_1Tanh rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/Tanh_1?
rnn/peephole_lstm_cell/mul_5Mul$rnn/peephole_lstm_cell/Sigmoid_2:y:0!rnn/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_5?
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!rnn/TensorArrayV2_1/element_shape?
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time?
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counter?
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:05rnn_peephole_lstm_cell_matmul_readvariableop_resource7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource6rnn_peephole_lstm_cell_biasadd_readvariableop_resource.rnn_peephole_lstm_cell_readvariableop_resource0rnn_peephole_lstm_cell_readvariableop_1_resource0rnn_peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
* 
bodyR
rnn_while_body_29314* 
condR
rnn_while_cond_29313*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
	rnn/while?
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    26
4rnn/TensorArrayV2Stack/TensorListStack/element_shape?
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2????????? *
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack?
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
rnn/strided_slice_3/stack?
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1?
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2?
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
rnn/strided_slice_3?
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/perm?
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2 2
rnn/transpose_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulrnn/strided_slice_3:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoid?
IdentityIdentitydense_3/Sigmoid:y:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_2/embedding_lookup.^rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp-^rnn/peephole_lstm_cell/MatMul/ReadVariableOp/^rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp&^rnn/peephole_lstm_cell/ReadVariableOp(^rnn/peephole_lstm_cell/ReadVariableOp_1(^rnn/peephole_lstm_cell/ReadVariableOp_2
^rnn/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2^
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp2\
,rnn/peephole_lstm_cell/MatMul/ReadVariableOp,rnn/peephole_lstm_cell/MatMul/ReadVariableOp2`
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp2N
%rnn/peephole_lstm_cell/ReadVariableOp%rnn/peephole_lstm_cell/ReadVariableOp2R
'rnn/peephole_lstm_cell/ReadVariableOp_1'rnn/peephole_lstm_cell/ReadVariableOp_12R
'rnn/peephole_lstm_cell/ReadVariableOp_2'rnn/peephole_lstm_cell/ReadVariableOp_22
	rnn/while	rnn/while:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?'
?
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_27910

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
readvariableop_resource
readvariableop_1_resource
readvariableop_2_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:????????? 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:????????? 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:????????? 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:????????? 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:????????? 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:????????? 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:????????? 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:????????? 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:????????? 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
mul_5?
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*d
_input_shapesS
Q:????????? :????????? :????????? ::::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
?*
?
while_body_28301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 while_peephole_lstm_cell_28325_0$
 while_peephole_lstm_cell_28327_0$
 while_peephole_lstm_cell_28329_0$
 while_peephole_lstm_cell_28331_0$
 while_peephole_lstm_cell_28333_0$
 while_peephole_lstm_cell_28335_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
while_peephole_lstm_cell_28325"
while_peephole_lstm_cell_28327"
while_peephole_lstm_cell_28329"
while_peephole_lstm_cell_28331"
while_peephole_lstm_cell_28333"
while_peephole_lstm_cell_28335??0while/peephole_lstm_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
0while/peephole_lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3 while_peephole_lstm_cell_28325_0 while_peephole_lstm_cell_28327_0 while_peephole_lstm_cell_28329_0 while_peephole_lstm_cell_28331_0 while_peephole_lstm_cell_28333_0 while_peephole_lstm_cell_28335_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_2791022
0while/peephole_lstm_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/peephole_lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/peephole_lstm_cell/StatefulPartitionedCall:output:11^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity9while/peephole_lstm_cell/StatefulPartitionedCall:output:21^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"B
while_peephole_lstm_cell_28325 while_peephole_lstm_cell_28325_0"B
while_peephole_lstm_cell_28327 while_peephole_lstm_cell_28327_0"B
while_peephole_lstm_cell_28329 while_peephole_lstm_cell_28329_0"B
while_peephole_lstm_cell_28331 while_peephole_lstm_cell_28331_0"B
while_peephole_lstm_cell_28333 while_peephole_lstm_cell_28333_0"B
while_peephole_lstm_cell_28335 while_peephole_lstm_cell_28335_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2d
0while/peephole_lstm_cell/StatefulPartitionedCall0while/peephole_lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
#__inference_rnn_layer_call_fn_30137

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_287962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????2 ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?p
?
>__inference_rnn_layer_call_and_return_conditional_losses_28796

inputs5
1peephole_lstm_cell_matmul_readvariableop_resource7
3peephole_lstm_cell_matmul_1_readvariableop_resource6
2peephole_lstm_cell_biasadd_readvariableop_resource.
*peephole_lstm_cell_readvariableop_resource0
,peephole_lstm_cell_readvariableop_1_resource0
,peephole_lstm_cell_readvariableop_2_resource
identity??)peephole_lstm_cell/BiasAdd/ReadVariableOp?(peephole_lstm_cell/MatMul/ReadVariableOp?*peephole_lstm_cell/MatMul_1/ReadVariableOp?!peephole_lstm_cell/ReadVariableOp?#peephole_lstm_cell/ReadVariableOp_1?#peephole_lstm_cell/ReadVariableOp_2?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2????????? 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_2?
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp?
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul?
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp?
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul_1?
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/add?
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOp?
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/BiasAddv
peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
peephole_lstm_cell/Const?
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim?
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
peephole_lstm_cell/split?
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp?
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul?
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_1?
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid?
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1?
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_1?
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_2?
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_1?
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_2?
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh?
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_3?
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_3?
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2?
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_4?
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_4?
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_2?
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh_1?
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_28694*
condR
while_cond_28693*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2 2
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????2 ::::::2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?d
?
while_body_29836
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_peephole_lstm_cell_matmul_readvariableop_resource_0?
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0>
:while_peephole_lstm_cell_biasadd_readvariableop_resource_06
2while_peephole_lstm_cell_readvariableop_resource_08
4while_peephole_lstm_cell_readvariableop_1_resource_08
4while_peephole_lstm_cell_readvariableop_2_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_peephole_lstm_cell_matmul_readvariableop_resource=
9while_peephole_lstm_cell_matmul_1_readvariableop_resource<
8while_peephole_lstm_cell_biasadd_readvariableop_resource4
0while_peephole_lstm_cell_readvariableop_resource6
2while_peephole_lstm_cell_readvariableop_1_resource6
2while_peephole_lstm_cell_readvariableop_2_resource??/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?.while/peephole_lstm_cell/MatMul/ReadVariableOp?0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?'while/peephole_lstm_cell/ReadVariableOp?)while/peephole_lstm_cell/ReadVariableOp_1?)while/peephole_lstm_cell/ReadVariableOp_2?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOp?
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/peephole_lstm_cell/MatMul?
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/peephole_lstm_cell/MatMul_1?
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/peephole_lstm_cell/add?
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/peephole_lstm_cell/BiasAdd?
while/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
while/peephole_lstm_cell/Const?
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim?
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2 
while/peephole_lstm_cell/split?
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp?
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/mul?
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_1?
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 while/peephole_lstm_cell/Sigmoid?
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1?
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_1?
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_2?
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_1?
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_2?
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/Tanh?
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_3?
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_3?
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2?
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_4?
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_4?
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_2?
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2!
while/peephole_lstm_cell/Tanh_1?
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?'
?
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_30662

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
readvariableop_resource
readvariableop_1_resource
readvariableop_2_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:????????? 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:????????? 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:????????? 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:????????? 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:????????? 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:????????? 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:????????? 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:????????? 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:????????? 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
mul_5?
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*d
_input_shapesS
Q:????????? :????????? :????????? ::::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?l
?
rnn_while_body_29314$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0A
=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0C
?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0B
>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0:
6rnn_while_peephole_lstm_cell_readvariableop_resource_0<
8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0<
8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor?
;rnn_while_peephole_lstm_cell_matmul_readvariableop_resourceA
=rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource@
<rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource8
4rnn_while_peephole_lstm_cell_readvariableop_resource:
6rnn_while_peephole_lstm_cell_readvariableop_1_resource:
6rnn_while_peephole_lstm_cell_readvariableop_2_resource??3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp?4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp?+rnn/while/peephole_lstm_cell/ReadVariableOp?-rnn/while/peephole_lstm_cell/ReadVariableOp_1?-rnn/while/peephole_lstm_cell/ReadVariableOp_2?
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2=
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02/
-rnn/while/TensorArrayV2Read/TensorListGetItem?
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype024
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp?
#rnn/while/peephole_lstm_cell/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:0:rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#rnn/while/peephole_lstm_cell/MatMul?
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype026
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp?
%rnn/while/peephole_lstm_cell/MatMul_1MatMulrnn_while_placeholder_2<rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%rnn/while/peephole_lstm_cell/MatMul_1?
 rnn/while/peephole_lstm_cell/addAddV2-rnn/while/peephole_lstm_cell/MatMul:product:0/rnn/while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2"
 rnn/while/peephole_lstm_cell/add?
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype025
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?
$rnn/while/peephole_lstm_cell/BiasAddBiasAdd$rnn/while/peephole_lstm_cell/add:z:0;rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$rnn/while/peephole_lstm_cell/BiasAdd?
"rnn/while/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"rnn/while/peephole_lstm_cell/Const?
,rnn/while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,rnn/while/peephole_lstm_cell/split/split_dim?
"rnn/while/peephole_lstm_cell/splitSplit5rnn/while/peephole_lstm_cell/split/split_dim:output:0-rnn/while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2$
"rnn/while/peephole_lstm_cell/split?
+rnn/while/peephole_lstm_cell/ReadVariableOpReadVariableOp6rnn_while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02-
+rnn/while/peephole_lstm_cell/ReadVariableOp?
 rnn/while/peephole_lstm_cell/mulMul3rnn/while/peephole_lstm_cell/ReadVariableOp:value:0rnn_while_placeholder_3*
T0*'
_output_shapes
:????????? 2"
 rnn/while/peephole_lstm_cell/mul?
"rnn/while/peephole_lstm_cell/add_1AddV2+rnn/while/peephole_lstm_cell/split:output:0$rnn/while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/add_1?
$rnn/while/peephole_lstm_cell/SigmoidSigmoid&rnn/while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2&
$rnn/while/peephole_lstm_cell/Sigmoid?
-rnn/while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02/
-rnn/while/peephole_lstm_cell/ReadVariableOp_1?
"rnn/while/peephole_lstm_cell/mul_1Mul5rnn/while/peephole_lstm_cell/ReadVariableOp_1:value:0rnn_while_placeholder_3*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_1?
"rnn/while/peephole_lstm_cell/add_2AddV2+rnn/while/peephole_lstm_cell/split:output:1&rnn/while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/add_2?
&rnn/while/peephole_lstm_cell/Sigmoid_1Sigmoid&rnn/while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2(
&rnn/while/peephole_lstm_cell/Sigmoid_1?
"rnn/while/peephole_lstm_cell/mul_2Mul*rnn/while/peephole_lstm_cell/Sigmoid_1:y:0rnn_while_placeholder_3*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_2?
!rnn/while/peephole_lstm_cell/TanhTanh+rnn/while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2#
!rnn/while/peephole_lstm_cell/Tanh?
"rnn/while/peephole_lstm_cell/mul_3Mul(rnn/while/peephole_lstm_cell/Sigmoid:y:0%rnn/while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_3?
"rnn/while/peephole_lstm_cell/add_3AddV2&rnn/while/peephole_lstm_cell/mul_2:z:0&rnn/while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/add_3?
-rnn/while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02/
-rnn/while/peephole_lstm_cell/ReadVariableOp_2?
"rnn/while/peephole_lstm_cell/mul_4Mul5rnn/while/peephole_lstm_cell/ReadVariableOp_2:value:0&rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_4?
"rnn/while/peephole_lstm_cell/add_4AddV2+rnn/while/peephole_lstm_cell/split:output:3&rnn/while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/add_4?
&rnn/while/peephole_lstm_cell/Sigmoid_2Sigmoid&rnn/while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2(
&rnn/while/peephole_lstm_cell/Sigmoid_2?
#rnn/while/peephole_lstm_cell/Tanh_1Tanh&rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2%
#rnn/while/peephole_lstm_cell/Tanh_1?
"rnn/while/peephole_lstm_cell/mul_5Mul*rnn/while/peephole_lstm_cell/Sigmoid_2:y:0'rnn/while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_5?
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder&rnn/while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype020
.rnn/while/TensorArrayV2Write/TensorListSetItemd
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add/yy
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn/while/addh
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add_1/y?
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn/while/add_1?
rnn/while/IdentityIdentityrnn/while/add_1:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity?
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations4^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_1?
rnn/while/Identity_2Identityrnn/while/add:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_2?
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_3?
rnn/while/Identity_4Identity&rnn/while/peephole_lstm_cell/mul_5:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
rnn/while/Identity_4?
rnn/while/Identity_5Identity&rnn/while/peephole_lstm_cell/add_3:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
rnn/while/Identity_5"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"~
<rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0"?
=rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"|
;rnn_while_peephole_lstm_cell_matmul_readvariableop_resource=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0"r
6rnn_while_peephole_lstm_cell_readvariableop_1_resource8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0"r
6rnn_while_peephole_lstm_cell_readvariableop_2_resource8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0"n
4rnn_while_peephole_lstm_cell_readvariableop_resource6rnn_while_peephole_lstm_cell_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"?
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2j
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2h
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2l
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp2Z
+rnn/while/peephole_lstm_cell/ReadVariableOp+rnn/while/peephole_lstm_cell/ReadVariableOp2^
-rnn/while/peephole_lstm_cell/ReadVariableOp_1-rnn/while/peephole_lstm_cell/ReadVariableOp_12^
-rnn/while/peephole_lstm_cell/ReadVariableOp_2-rnn/while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
l
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_28603

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????2 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:?????????2 :S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
??
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29619

inputs&
"embedding_2_embedding_lookup_294279
5rnn_peephole_lstm_cell_matmul_readvariableop_resource;
7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource:
6rnn_peephole_lstm_cell_biasadd_readvariableop_resource2
.rnn_peephole_lstm_cell_readvariableop_resource4
0rnn_peephole_lstm_cell_readvariableop_1_resource4
0rnn_peephole_lstm_cell_readvariableop_2_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_2/embedding_lookup?-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp?,rnn/peephole_lstm_cell/MatMul/ReadVariableOp?.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp?%rnn/peephole_lstm_cell/ReadVariableOp?'rnn/peephole_lstm_cell/ReadVariableOp_1?'rnn/peephole_lstm_cell/ReadVariableOp_2?	rnn/whileu
embedding_2/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????22
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_29427embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_2/embedding_lookup/29427*+
_output_shapes
:?????????2 *
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/29427*+
_output_shapes
:?????????2 2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2 2)
'embedding_2/embedding_lookup/Identity_1?
spatial_dropout1d_2/IdentityIdentity0embedding_2/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????2 2
spatial_dropout1d_2/Identityk
	rnn/ShapeShape%spatial_dropout1d_2/Identity:output:0*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack?
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1?
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2?
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_sliced
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessj
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros/packed/1?
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const?
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
	rnn/zerosh
rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros_1/mul/y?
rnn/zeros_1/mulMulrnn/strided_slice:output:0rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/mulk
rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
rnn/zeros_1/Less/y
rnn/zeros_1/LessLessrnn/zeros_1/mul:z:0rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/Lessn
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
rnn/zeros_1/packed/1?
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_1/packedk
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_1/Const?
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
rnn/zeros_1}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm?
rnn/transpose	Transpose%spatial_dropout1d_2/Identity:output:0rnn/transpose/perm:output:0*
T0*+
_output_shapes
:2????????? 2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1?
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack?
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1?
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2?
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1?
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
rnn/TensorArrayV2/element_shape?
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2?
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor?
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack?
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1?
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2?
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
rnn/strided_slice_2?
,rnn/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp5rnn_peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02.
,rnn/peephole_lstm_cell/MatMul/ReadVariableOp?
rnn/peephole_lstm_cell/MatMulMatMulrnn/strided_slice_2:output:04rnn/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
rnn/peephole_lstm_cell/MatMul?
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype020
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp?
rnn/peephole_lstm_cell/MatMul_1MatMulrnn/zeros:output:06rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
rnn/peephole_lstm_cell/MatMul_1?
rnn/peephole_lstm_cell/addAddV2'rnn/peephole_lstm_cell/MatMul:product:0)rnn/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
rnn/peephole_lstm_cell/add?
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6rnn_peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp?
rnn/peephole_lstm_cell/BiasAddBiasAddrnn/peephole_lstm_cell/add:z:05rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
rnn/peephole_lstm_cell/BiasAdd~
rnn/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/peephole_lstm_cell/Const?
&rnn/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&rnn/peephole_lstm_cell/split/split_dim?
rnn/peephole_lstm_cell/splitSplit/rnn/peephole_lstm_cell/split/split_dim:output:0'rnn/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
rnn/peephole_lstm_cell/split?
%rnn/peephole_lstm_cell/ReadVariableOpReadVariableOp.rnn_peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02'
%rnn/peephole_lstm_cell/ReadVariableOp?
rnn/peephole_lstm_cell/mulMul-rnn/peephole_lstm_cell/ReadVariableOp:value:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul?
rnn/peephole_lstm_cell/add_1AddV2%rnn/peephole_lstm_cell/split:output:0rnn/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/add_1?
rnn/peephole_lstm_cell/SigmoidSigmoid rnn/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2 
rnn/peephole_lstm_cell/Sigmoid?
'rnn/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp0rnn_peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'rnn/peephole_lstm_cell/ReadVariableOp_1?
rnn/peephole_lstm_cell/mul_1Mul/rnn/peephole_lstm_cell/ReadVariableOp_1:value:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_1?
rnn/peephole_lstm_cell/add_2AddV2%rnn/peephole_lstm_cell/split:output:1 rnn/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/add_2?
 rnn/peephole_lstm_cell/Sigmoid_1Sigmoid rnn/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2"
 rnn/peephole_lstm_cell/Sigmoid_1?
rnn/peephole_lstm_cell/mul_2Mul$rnn/peephole_lstm_cell/Sigmoid_1:y:0rnn/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_2?
rnn/peephole_lstm_cell/TanhTanh%rnn/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/Tanh?
rnn/peephole_lstm_cell/mul_3Mul"rnn/peephole_lstm_cell/Sigmoid:y:0rnn/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_3?
rnn/peephole_lstm_cell/add_3AddV2 rnn/peephole_lstm_cell/mul_2:z:0 rnn/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/add_3?
'rnn/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp0rnn_peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02)
'rnn/peephole_lstm_cell/ReadVariableOp_2?
rnn/peephole_lstm_cell/mul_4Mul/rnn/peephole_lstm_cell/ReadVariableOp_2:value:0 rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_4?
rnn/peephole_lstm_cell/add_4AddV2%rnn/peephole_lstm_cell/split:output:3 rnn/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/add_4?
 rnn/peephole_lstm_cell/Sigmoid_2Sigmoid rnn/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2"
 rnn/peephole_lstm_cell/Sigmoid_2?
rnn/peephole_lstm_cell/Tanh_1Tanh rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/Tanh_1?
rnn/peephole_lstm_cell/mul_5Mul$rnn/peephole_lstm_cell/Sigmoid_2:y:0!rnn/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
rnn/peephole_lstm_cell/mul_5?
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2#
!rnn/TensorArrayV2_1/element_shape?
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time?
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counter?
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:05rnn_peephole_lstm_cell_matmul_readvariableop_resource7rnn_peephole_lstm_cell_matmul_1_readvariableop_resource6rnn_peephole_lstm_cell_biasadd_readvariableop_resource.rnn_peephole_lstm_cell_readvariableop_resource0rnn_peephole_lstm_cell_readvariableop_1_resource0rnn_peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
* 
bodyR
rnn_while_body_29510* 
condR
rnn_while_cond_29509*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
	rnn/while?
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    26
4rnn/TensorArrayV2Stack/TensorListStack/element_shape?
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2????????? *
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack?
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
rnn/strided_slice_3/stack?
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1?
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2?
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
rnn/strided_slice_3?
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/perm?
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2 2
rnn/transpose_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulrnn/strided_slice_3:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoid?
IdentityIdentitydense_3/Sigmoid:y:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_2/embedding_lookup.^rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp-^rnn/peephole_lstm_cell/MatMul/ReadVariableOp/^rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp&^rnn/peephole_lstm_cell/ReadVariableOp(^rnn/peephole_lstm_cell/ReadVariableOp_1(^rnn/peephole_lstm_cell/ReadVariableOp_2
^rnn/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2^
-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp-rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp2\
,rnn/peephole_lstm_cell/MatMul/ReadVariableOp,rnn/peephole_lstm_cell/MatMul/ReadVariableOp2`
.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp.rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp2N
%rnn/peephole_lstm_cell/ReadVariableOp%rnn/peephole_lstm_cell/ReadVariableOp2R
'rnn/peephole_lstm_cell/ReadVariableOp_1'rnn/peephole_lstm_cell/ReadVariableOp_12R
'rnn/peephole_lstm_cell/ReadVariableOp_2'rnn/peephole_lstm_cell/ReadVariableOp_22
	rnn/while	rnn/while:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_29210
embedding_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_277592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:?????????2
+
_user_specified_nameembedding_2_input
?p
?
>__inference_rnn_layer_call_and_return_conditional_losses_30120

inputs5
1peephole_lstm_cell_matmul_readvariableop_resource7
3peephole_lstm_cell_matmul_1_readvariableop_resource6
2peephole_lstm_cell_biasadd_readvariableop_resource.
*peephole_lstm_cell_readvariableop_resource0
,peephole_lstm_cell_readvariableop_1_resource0
,peephole_lstm_cell_readvariableop_2_resource
identity??)peephole_lstm_cell/BiasAdd/ReadVariableOp?(peephole_lstm_cell/MatMul/ReadVariableOp?*peephole_lstm_cell/MatMul_1/ReadVariableOp?!peephole_lstm_cell/ReadVariableOp?#peephole_lstm_cell/ReadVariableOp_1?#peephole_lstm_cell/ReadVariableOp_2?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2????????? 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_2?
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp?
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul?
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp?
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul_1?
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/add?
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOp?
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/BiasAddv
peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
peephole_lstm_cell/Const?
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim?
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
peephole_lstm_cell/split?
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp?
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul?
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_1?
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid?
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1?
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_1?
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_2?
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_1?
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_2?
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh?
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_3?
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_3?
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2?
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_4?
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_4?
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_2?
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh_1?
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_30018*
condR
while_cond_30017*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2 2
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????2 ::::::2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?
q
+__inference_embedding_2_layer_call_fn_29682

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_285652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_29179
embedding_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_291582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:?????????2
+
_user_specified_nameembedding_2_input
?
m
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29741

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????2 2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????2 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2 :S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?F
?
>__inference_rnn_layer_call_and_return_conditional_losses_28536

inputs
peephole_lstm_cell_28437
peephole_lstm_cell_28439
peephole_lstm_cell_28441
peephole_lstm_cell_28443
peephole_lstm_cell_28445
peephole_lstm_cell_28447
identity??*peephole_lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_2?
*peephole_lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0peephole_lstm_cell_28437peephole_lstm_cell_28439peephole_lstm_cell_28441peephole_lstm_cell_28443peephole_lstm_cell_28445peephole_lstm_cell_28447*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_279552,
*peephole_lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0peephole_lstm_cell_28437peephole_lstm_cell_28439peephole_lstm_cell_28441peephole_lstm_cell_28443peephole_lstm_cell_28445peephole_lstm_cell_28447*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_28456*
condR
while_cond_28455*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1?
IdentityIdentitystrided_slice_3:output:0+^peephole_lstm_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????????????? ::::::2X
*peephole_lstm_cell/StatefulPartitionedCall*peephole_lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
m
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_27812

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_29665

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_291582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?'
?
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_30617

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
readvariableop_resource
readvariableop_1_resource
readvariableop_2_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:????????? 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:????????? 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:????????? 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:????????? 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:????????? 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:????????? 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:????????? 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:????????? 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:????????? 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
mul_5?
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*d
_input_shapesS
Q:????????? :????????? :????????? ::::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?	
?
while_cond_30017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_30017___redundant_placeholder03
/while_while_cond_30017___redundant_placeholder13
/while_while_cond_30017___redundant_placeholder23
/while_while_cond_30017___redundant_placeholder33
/while_while_cond_30017___redundant_placeholder43
/while_while_cond_30017___redundant_placeholder53
/while_while_cond_30017___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?k
?
!__inference__traced_restore_30885
file_prefix+
'assignvariableop_embedding_2_embeddings%
!assignvariableop_1_dense_3_kernel#
assignvariableop_2_dense_3_bias
assignvariableop_3_sgd_iter 
assignvariableop_4_sgd_decay(
$assignvariableop_5_sgd_learning_rate#
assignvariableop_6_sgd_momentum4
0assignvariableop_7_rnn_peephole_lstm_cell_kernel>
:assignvariableop_8_rnn_peephole_lstm_cell_recurrent_kernel2
.assignvariableop_9_rnn_peephole_lstm_cell_biasJ
Fassignvariableop_10_rnn_peephole_lstm_cell_input_gate_peephole_weightsK
Gassignvariableop_11_rnn_peephole_lstm_cell_forget_gate_peephole_weightsK
Gassignvariableop_12_rnn_peephole_lstm_cell_output_gate_peephole_weights
assignvariableop_13_total
assignvariableop_14_count;
7assignvariableop_15_sgd_embedding_2_embeddings_momentum3
/assignvariableop_16_sgd_dense_3_kernel_momentum1
-assignvariableop_17_sgd_dense_3_bias_momentumB
>assignvariableop_18_sgd_rnn_peephole_lstm_cell_kernel_momentumL
Hassignvariableop_19_sgd_rnn_peephole_lstm_cell_recurrent_kernel_momentum@
<assignvariableop_20_sgd_rnn_peephole_lstm_cell_bias_momentumW
Sassignvariableop_21_sgd_rnn_peephole_lstm_cell_input_gate_peephole_weights_momentumX
Tassignvariableop_22_sgd_rnn_peephole_lstm_cell_forget_gate_peephole_weights_momentumX
Tassignvariableop_23_sgd_rnn_peephole_lstm_cell_output_gate_peephole_weights_momentum
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_2_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_3_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_3_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_sgd_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_decayIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_sgd_learning_rateIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_momentumIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp0assignvariableop_7_rnn_peephole_lstm_cell_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp:assignvariableop_8_rnn_peephole_lstm_cell_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_rnn_peephole_lstm_cell_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpFassignvariableop_10_rnn_peephole_lstm_cell_input_gate_peephole_weightsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpGassignvariableop_11_rnn_peephole_lstm_cell_forget_gate_peephole_weightsIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpGassignvariableop_12_rnn_peephole_lstm_cell_output_gate_peephole_weightsIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp7assignvariableop_15_sgd_embedding_2_embeddings_momentumIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_sgd_dense_3_kernel_momentumIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp-assignvariableop_17_sgd_dense_3_bias_momentumIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp>assignvariableop_18_sgd_rnn_peephole_lstm_cell_kernel_momentumIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpHassignvariableop_19_sgd_rnn_peephole_lstm_cell_recurrent_kernel_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp<assignvariableop_20_sgd_rnn_peephole_lstm_cell_bias_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpSassignvariableop_21_sgd_rnn_peephole_lstm_cell_input_gate_peephole_weights_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpTassignvariableop_22_sgd_rnn_peephole_lstm_cell_forget_gate_peephole_weights_momentumIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpTassignvariableop_23_sgd_rnn_peephole_lstm_cell_output_gate_peephole_weights_momentumIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24?
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
m
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29704

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
|
'__inference_dense_3_layer_call_fn_30572

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_290372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
while_cond_28693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_28693___redundant_placeholder03
/while_while_cond_28693___redundant_placeholder13
/while_while_cond_28693___redundant_placeholder23
/while_while_cond_28693___redundant_placeholder33
/while_while_cond_28693___redundant_placeholder43
/while_while_cond_28693___redundant_placeholder53
/while_while_cond_28693___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?p
?
>__inference_rnn_layer_call_and_return_conditional_losses_30336
inputs_05
1peephole_lstm_cell_matmul_readvariableop_resource7
3peephole_lstm_cell_matmul_1_readvariableop_resource6
2peephole_lstm_cell_biasadd_readvariableop_resource.
*peephole_lstm_cell_readvariableop_resource0
,peephole_lstm_cell_readvariableop_1_resource0
,peephole_lstm_cell_readvariableop_2_resource
identity??)peephole_lstm_cell/BiasAdd/ReadVariableOp?(peephole_lstm_cell/MatMul/ReadVariableOp?*peephole_lstm_cell/MatMul_1/ReadVariableOp?!peephole_lstm_cell/ReadVariableOp?#peephole_lstm_cell/ReadVariableOp_1?#peephole_lstm_cell/ReadVariableOp_2?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_2?
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp?
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul?
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp?
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul_1?
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/add?
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOp?
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/BiasAddv
peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
peephole_lstm_cell/Const?
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim?
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
peephole_lstm_cell/split?
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp?
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul?
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_1?
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid?
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1?
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_1?
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_2?
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_1?
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_2?
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh?
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_3?
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_3?
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2?
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_4?
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_4?
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_2?
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh_1?
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_30234*
condR
while_cond_30233*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????????????? ::::::2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?p
?
>__inference_rnn_layer_call_and_return_conditional_losses_28978

inputs5
1peephole_lstm_cell_matmul_readvariableop_resource7
3peephole_lstm_cell_matmul_1_readvariableop_resource6
2peephole_lstm_cell_biasadd_readvariableop_resource.
*peephole_lstm_cell_readvariableop_resource0
,peephole_lstm_cell_readvariableop_1_resource0
,peephole_lstm_cell_readvariableop_2_resource
identity??)peephole_lstm_cell/BiasAdd/ReadVariableOp?(peephole_lstm_cell/MatMul/ReadVariableOp?*peephole_lstm_cell/MatMul_1/ReadVariableOp?!peephole_lstm_cell/ReadVariableOp?#peephole_lstm_cell/ReadVariableOp_1?#peephole_lstm_cell/ReadVariableOp_2?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2????????? 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_2?
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp?
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul?
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp?
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul_1?
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/add?
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOp?
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/BiasAddv
peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
peephole_lstm_cell/Const?
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim?
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
peephole_lstm_cell/split?
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp?
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul?
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_1?
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid?
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1?
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_1?
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_2?
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_1?
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_2?
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh?
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_3?
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_3?
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2?
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_4?
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_4?
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_2?
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh_1?
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_28876*
condR
while_cond_28875*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2 2
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????2 ::::::2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?
l
3__inference_spatial_dropout1d_2_layer_call_fn_29751

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_285982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?	
?
while_cond_28455
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_28455___redundant_placeholder03
/while_while_cond_28455___redundant_placeholder13
/while_while_cond_28455___redundant_placeholder23
/while_while_cond_28455___redundant_placeholder33
/while_while_cond_28455___redundant_placeholder43
/while_while_cond_28455___redundant_placeholder53
/while_while_cond_28455___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?p
?
>__inference_rnn_layer_call_and_return_conditional_losses_29938

inputs5
1peephole_lstm_cell_matmul_readvariableop_resource7
3peephole_lstm_cell_matmul_1_readvariableop_resource6
2peephole_lstm_cell_biasadd_readvariableop_resource.
*peephole_lstm_cell_readvariableop_resource0
,peephole_lstm_cell_readvariableop_1_resource0
,peephole_lstm_cell_readvariableop_2_resource
identity??)peephole_lstm_cell/BiasAdd/ReadVariableOp?(peephole_lstm_cell/MatMul/ReadVariableOp?*peephole_lstm_cell/MatMul_1/ReadVariableOp?!peephole_lstm_cell/ReadVariableOp?#peephole_lstm_cell/ReadVariableOp_1?#peephole_lstm_cell/ReadVariableOp_2?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:2????????? 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_2?
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp?
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul?
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp?
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul_1?
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/add?
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOp?
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/BiasAddv
peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
peephole_lstm_cell/Const?
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim?
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
peephole_lstm_cell/split?
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp?
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul?
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_1?
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid?
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1?
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_1?
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_2?
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_1?
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_2?
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh?
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_3?
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_3?
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2?
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_4?
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_4?
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_2?
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh_1?
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_29836*
condR
while_cond_29835*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2 2
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????2 ::::::2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?
l
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29709

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
!sequential_2_rnn_while_body_27650>
:sequential_2_rnn_while_sequential_2_rnn_while_loop_counterD
@sequential_2_rnn_while_sequential_2_rnn_while_maximum_iterations&
"sequential_2_rnn_while_placeholder(
$sequential_2_rnn_while_placeholder_1(
$sequential_2_rnn_while_placeholder_2(
$sequential_2_rnn_while_placeholder_3=
9sequential_2_rnn_while_sequential_2_rnn_strided_slice_1_0y
usequential_2_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_rnn_tensorarrayunstack_tensorlistfromtensor_0N
Jsequential_2_rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0P
Lsequential_2_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0O
Ksequential_2_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0G
Csequential_2_rnn_while_peephole_lstm_cell_readvariableop_resource_0I
Esequential_2_rnn_while_peephole_lstm_cell_readvariableop_1_resource_0I
Esequential_2_rnn_while_peephole_lstm_cell_readvariableop_2_resource_0#
sequential_2_rnn_while_identity%
!sequential_2_rnn_while_identity_1%
!sequential_2_rnn_while_identity_2%
!sequential_2_rnn_while_identity_3%
!sequential_2_rnn_while_identity_4%
!sequential_2_rnn_while_identity_5;
7sequential_2_rnn_while_sequential_2_rnn_strided_slice_1w
ssequential_2_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_rnn_tensorarrayunstack_tensorlistfromtensorL
Hsequential_2_rnn_while_peephole_lstm_cell_matmul_readvariableop_resourceN
Jsequential_2_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resourceM
Isequential_2_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resourceE
Asequential_2_rnn_while_peephole_lstm_cell_readvariableop_resourceG
Csequential_2_rnn_while_peephole_lstm_cell_readvariableop_1_resourceG
Csequential_2_rnn_while_peephole_lstm_cell_readvariableop_2_resource??@sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp??sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp?Asequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp?8sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp?:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1?:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2?
Hsequential_2/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2J
Hsequential_2/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
:sequential_2/rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemusequential_2_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_rnn_tensorarrayunstack_tensorlistfromtensor_0"sequential_2_rnn_while_placeholderQsequential_2/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02<
:sequential_2/rnn/while/TensorArrayV2Read/TensorListGetItem?
?sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOpJsequential_2_rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02A
?sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp?
0sequential_2/rnn/while/peephole_lstm_cell/MatMulMatMulAsequential_2/rnn/while/TensorArrayV2Read/TensorListGetItem:item:0Gsequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_2/rnn/while/peephole_lstm_cell/MatMul?
Asequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOpLsequential_2_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02C
Asequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp?
2sequential_2/rnn/while/peephole_lstm_cell/MatMul_1MatMul$sequential_2_rnn_while_placeholder_2Isequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????24
2sequential_2/rnn/while/peephole_lstm_cell/MatMul_1?
-sequential_2/rnn/while/peephole_lstm_cell/addAddV2:sequential_2/rnn/while/peephole_lstm_cell/MatMul:product:0<sequential_2/rnn/while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2/
-sequential_2/rnn/while/peephole_lstm_cell/add?
@sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOpKsequential_2_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02B
@sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?
1sequential_2/rnn/while/peephole_lstm_cell/BiasAddBiasAdd1sequential_2/rnn/while/peephole_lstm_cell/add:z:0Hsequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????23
1sequential_2/rnn/while/peephole_lstm_cell/BiasAdd?
/sequential_2/rnn/while/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_2/rnn/while/peephole_lstm_cell/Const?
9sequential_2/rnn/while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_2/rnn/while/peephole_lstm_cell/split/split_dim?
/sequential_2/rnn/while/peephole_lstm_cell/splitSplitBsequential_2/rnn/while/peephole_lstm_cell/split/split_dim:output:0:sequential_2/rnn/while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split21
/sequential_2/rnn/while/peephole_lstm_cell/split?
8sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOpReadVariableOpCsequential_2_rnn_while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02:
8sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp?
-sequential_2/rnn/while/peephole_lstm_cell/mulMul@sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp:value:0$sequential_2_rnn_while_placeholder_3*
T0*'
_output_shapes
:????????? 2/
-sequential_2/rnn/while/peephole_lstm_cell/mul?
/sequential_2/rnn/while/peephole_lstm_cell/add_1AddV28sequential_2/rnn/while/peephole_lstm_cell/split:output:01sequential_2/rnn/while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 21
/sequential_2/rnn/while/peephole_lstm_cell/add_1?
1sequential_2/rnn/while/peephole_lstm_cell/SigmoidSigmoid3sequential_2/rnn/while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 23
1sequential_2/rnn/while/peephole_lstm_cell/Sigmoid?
:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOpEsequential_2_rnn_while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02<
:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1?
/sequential_2/rnn/while/peephole_lstm_cell/mul_1MulBsequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1:value:0$sequential_2_rnn_while_placeholder_3*
T0*'
_output_shapes
:????????? 21
/sequential_2/rnn/while/peephole_lstm_cell/mul_1?
/sequential_2/rnn/while/peephole_lstm_cell/add_2AddV28sequential_2/rnn/while/peephole_lstm_cell/split:output:13sequential_2/rnn/while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 21
/sequential_2/rnn/while/peephole_lstm_cell/add_2?
3sequential_2/rnn/while/peephole_lstm_cell/Sigmoid_1Sigmoid3sequential_2/rnn/while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 25
3sequential_2/rnn/while/peephole_lstm_cell/Sigmoid_1?
/sequential_2/rnn/while/peephole_lstm_cell/mul_2Mul7sequential_2/rnn/while/peephole_lstm_cell/Sigmoid_1:y:0$sequential_2_rnn_while_placeholder_3*
T0*'
_output_shapes
:????????? 21
/sequential_2/rnn/while/peephole_lstm_cell/mul_2?
.sequential_2/rnn/while/peephole_lstm_cell/TanhTanh8sequential_2/rnn/while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 20
.sequential_2/rnn/while/peephole_lstm_cell/Tanh?
/sequential_2/rnn/while/peephole_lstm_cell/mul_3Mul5sequential_2/rnn/while/peephole_lstm_cell/Sigmoid:y:02sequential_2/rnn/while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 21
/sequential_2/rnn/while/peephole_lstm_cell/mul_3?
/sequential_2/rnn/while/peephole_lstm_cell/add_3AddV23sequential_2/rnn/while/peephole_lstm_cell/mul_2:z:03sequential_2/rnn/while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 21
/sequential_2/rnn/while/peephole_lstm_cell/add_3?
:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOpEsequential_2_rnn_while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02<
:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2?
/sequential_2/rnn/while/peephole_lstm_cell/mul_4MulBsequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2:value:03sequential_2/rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 21
/sequential_2/rnn/while/peephole_lstm_cell/mul_4?
/sequential_2/rnn/while/peephole_lstm_cell/add_4AddV28sequential_2/rnn/while/peephole_lstm_cell/split:output:33sequential_2/rnn/while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 21
/sequential_2/rnn/while/peephole_lstm_cell/add_4?
3sequential_2/rnn/while/peephole_lstm_cell/Sigmoid_2Sigmoid3sequential_2/rnn/while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 25
3sequential_2/rnn/while/peephole_lstm_cell/Sigmoid_2?
0sequential_2/rnn/while/peephole_lstm_cell/Tanh_1Tanh3sequential_2/rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 22
0sequential_2/rnn/while/peephole_lstm_cell/Tanh_1?
/sequential_2/rnn/while/peephole_lstm_cell/mul_5Mul7sequential_2/rnn/while/peephole_lstm_cell/Sigmoid_2:y:04sequential_2/rnn/while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 21
/sequential_2/rnn/while/peephole_lstm_cell/mul_5?
;sequential_2/rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$sequential_2_rnn_while_placeholder_1"sequential_2_rnn_while_placeholder3sequential_2/rnn/while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02=
;sequential_2/rnn/while/TensorArrayV2Write/TensorListSetItem~
sequential_2/rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential_2/rnn/while/add/y?
sequential_2/rnn/while/addAddV2"sequential_2_rnn_while_placeholder%sequential_2/rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_2/rnn/while/add?
sequential_2/rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2 
sequential_2/rnn/while/add_1/y?
sequential_2/rnn/while/add_1AddV2:sequential_2_rnn_while_sequential_2_rnn_while_loop_counter'sequential_2/rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential_2/rnn/while/add_1?
sequential_2/rnn/while/IdentityIdentity sequential_2/rnn/while/add_1:z:0A^sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp@^sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpB^sequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp9^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2!
sequential_2/rnn/while/Identity?
!sequential_2/rnn/while/Identity_1Identity@sequential_2_rnn_while_sequential_2_rnn_while_maximum_iterationsA^sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp@^sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpB^sequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp9^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2#
!sequential_2/rnn/while/Identity_1?
!sequential_2/rnn/while/Identity_2Identitysequential_2/rnn/while/add:z:0A^sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp@^sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpB^sequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp9^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2#
!sequential_2/rnn/while/Identity_2?
!sequential_2/rnn/while/Identity_3IdentityKsequential_2/rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0A^sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp@^sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpB^sequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp9^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2#
!sequential_2/rnn/while/Identity_3?
!sequential_2/rnn/while/Identity_4Identity3sequential_2/rnn/while/peephole_lstm_cell/mul_5:z:0A^sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp@^sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpB^sequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp9^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2#
!sequential_2/rnn/while/Identity_4?
!sequential_2/rnn/while/Identity_5Identity3sequential_2/rnn/while/peephole_lstm_cell/add_3:z:0A^sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp@^sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpB^sequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp9^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1;^sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2#
!sequential_2/rnn/while/Identity_5"K
sequential_2_rnn_while_identity(sequential_2/rnn/while/Identity:output:0"O
!sequential_2_rnn_while_identity_1*sequential_2/rnn/while/Identity_1:output:0"O
!sequential_2_rnn_while_identity_2*sequential_2/rnn/while/Identity_2:output:0"O
!sequential_2_rnn_while_identity_3*sequential_2/rnn/while/Identity_3:output:0"O
!sequential_2_rnn_while_identity_4*sequential_2/rnn/while/Identity_4:output:0"O
!sequential_2_rnn_while_identity_5*sequential_2/rnn/while/Identity_5:output:0"?
Isequential_2_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resourceKsequential_2_rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0"?
Jsequential_2_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resourceLsequential_2_rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"?
Hsequential_2_rnn_while_peephole_lstm_cell_matmul_readvariableop_resourceJsequential_2_rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0"?
Csequential_2_rnn_while_peephole_lstm_cell_readvariableop_1_resourceEsequential_2_rnn_while_peephole_lstm_cell_readvariableop_1_resource_0"?
Csequential_2_rnn_while_peephole_lstm_cell_readvariableop_2_resourceEsequential_2_rnn_while_peephole_lstm_cell_readvariableop_2_resource_0"?
Asequential_2_rnn_while_peephole_lstm_cell_readvariableop_resourceCsequential_2_rnn_while_peephole_lstm_cell_readvariableop_resource_0"t
7sequential_2_rnn_while_sequential_2_rnn_strided_slice_19sequential_2_rnn_while_sequential_2_rnn_strided_slice_1_0"?
ssequential_2_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_rnn_tensorarrayunstack_tensorlistfromtensorusequential_2_rnn_while_tensorarrayv2read_tensorlistgetitem_sequential_2_rnn_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2?
@sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp@sequential_2/rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2?
?sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp?sequential_2/rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2?
Asequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOpAsequential_2/rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp2t
8sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp8sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp2x
:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_1:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_12x
:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2:sequential_2/rnn/while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_2_layer_call_fn_29642

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_291092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
#__inference_rnn_layer_call_fn_30535
inputs_0
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_283812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????????????? ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?
l
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_27822

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_29130
embedding_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallembedding_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_291092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:?????????2
+
_user_specified_nameembedding_2_input
?
l
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29746

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????2 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????2 2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:?????????2 :S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
د
?	
 __inference__wrapped_model_27759
embedding_2_input3
/sequential_2_embedding_2_embedding_lookup_27567F
Bsequential_2_rnn_peephole_lstm_cell_matmul_readvariableop_resourceH
Dsequential_2_rnn_peephole_lstm_cell_matmul_1_readvariableop_resourceG
Csequential_2_rnn_peephole_lstm_cell_biasadd_readvariableop_resource?
;sequential_2_rnn_peephole_lstm_cell_readvariableop_resourceA
=sequential_2_rnn_peephole_lstm_cell_readvariableop_1_resourceA
=sequential_2_rnn_peephole_lstm_cell_readvariableop_2_resource7
3sequential_2_dense_3_matmul_readvariableop_resource8
4sequential_2_dense_3_biasadd_readvariableop_resource
identity??+sequential_2/dense_3/BiasAdd/ReadVariableOp?*sequential_2/dense_3/MatMul/ReadVariableOp?)sequential_2/embedding_2/embedding_lookup?:sequential_2/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp?9sequential_2/rnn/peephole_lstm_cell/MatMul/ReadVariableOp?;sequential_2/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp?2sequential_2/rnn/peephole_lstm_cell/ReadVariableOp?4sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_1?4sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_2?sequential_2/rnn/while?
sequential_2/embedding_2/CastCastembedding_2_input*

DstT0*

SrcT0*'
_output_shapes
:?????????22
sequential_2/embedding_2/Cast?
)sequential_2/embedding_2/embedding_lookupResourceGather/sequential_2_embedding_2_embedding_lookup_27567!sequential_2/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@sequential_2/embedding_2/embedding_lookup/27567*+
_output_shapes
:?????????2 *
dtype02+
)sequential_2/embedding_2/embedding_lookup?
2sequential_2/embedding_2/embedding_lookup/IdentityIdentity2sequential_2/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@sequential_2/embedding_2/embedding_lookup/27567*+
_output_shapes
:?????????2 24
2sequential_2/embedding_2/embedding_lookup/Identity?
4sequential_2/embedding_2/embedding_lookup/Identity_1Identity;sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2 26
4sequential_2/embedding_2/embedding_lookup/Identity_1?
)sequential_2/spatial_dropout1d_2/IdentityIdentity=sequential_2/embedding_2/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????2 2+
)sequential_2/spatial_dropout1d_2/Identity?
sequential_2/rnn/ShapeShape2sequential_2/spatial_dropout1d_2/Identity:output:0*
T0*
_output_shapes
:2
sequential_2/rnn/Shape?
$sequential_2/rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_2/rnn/strided_slice/stack?
&sequential_2/rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_2/rnn/strided_slice/stack_1?
&sequential_2/rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_2/rnn/strided_slice/stack_2?
sequential_2/rnn/strided_sliceStridedSlicesequential_2/rnn/Shape:output:0-sequential_2/rnn/strided_slice/stack:output:0/sequential_2/rnn/strided_slice/stack_1:output:0/sequential_2/rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sequential_2/rnn/strided_slice~
sequential_2/rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_2/rnn/zeros/mul/y?
sequential_2/rnn/zeros/mulMul'sequential_2/rnn/strided_slice:output:0%sequential_2/rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_2/rnn/zeros/mul?
sequential_2/rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
sequential_2/rnn/zeros/Less/y?
sequential_2/rnn/zeros/LessLesssequential_2/rnn/zeros/mul:z:0&sequential_2/rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential_2/rnn/zeros/Less?
sequential_2/rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_2/rnn/zeros/packed/1?
sequential_2/rnn/zeros/packedPack'sequential_2/rnn/strided_slice:output:0(sequential_2/rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential_2/rnn/zeros/packed?
sequential_2/rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_2/rnn/zeros/Const?
sequential_2/rnn/zerosFill&sequential_2/rnn/zeros/packed:output:0%sequential_2/rnn/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
sequential_2/rnn/zeros?
sequential_2/rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential_2/rnn/zeros_1/mul/y?
sequential_2/rnn/zeros_1/mulMul'sequential_2/rnn/strided_slice:output:0'sequential_2/rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_2/rnn/zeros_1/mul?
sequential_2/rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
sequential_2/rnn/zeros_1/Less/y?
sequential_2/rnn/zeros_1/LessLess sequential_2/rnn/zeros_1/mul:z:0(sequential_2/rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential_2/rnn/zeros_1/Less?
!sequential_2/rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_2/rnn/zeros_1/packed/1?
sequential_2/rnn/zeros_1/packedPack'sequential_2/rnn/strided_slice:output:0*sequential_2/rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2!
sequential_2/rnn/zeros_1/packed?
sequential_2/rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
sequential_2/rnn/zeros_1/Const?
sequential_2/rnn/zeros_1Fill(sequential_2/rnn/zeros_1/packed:output:0'sequential_2/rnn/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
sequential_2/rnn/zeros_1?
sequential_2/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential_2/rnn/transpose/perm?
sequential_2/rnn/transpose	Transpose2sequential_2/spatial_dropout1d_2/Identity:output:0(sequential_2/rnn/transpose/perm:output:0*
T0*+
_output_shapes
:2????????? 2
sequential_2/rnn/transpose?
sequential_2/rnn/Shape_1Shapesequential_2/rnn/transpose:y:0*
T0*
_output_shapes
:2
sequential_2/rnn/Shape_1?
&sequential_2/rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/rnn/strided_slice_1/stack?
(sequential_2/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_2/rnn/strided_slice_1/stack_1?
(sequential_2/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_2/rnn/strided_slice_1/stack_2?
 sequential_2/rnn/strided_slice_1StridedSlice!sequential_2/rnn/Shape_1:output:0/sequential_2/rnn/strided_slice_1/stack:output:01sequential_2/rnn/strided_slice_1/stack_1:output:01sequential_2/rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 sequential_2/rnn/strided_slice_1?
,sequential_2/rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,sequential_2/rnn/TensorArrayV2/element_shape?
sequential_2/rnn/TensorArrayV2TensorListReserve5sequential_2/rnn/TensorArrayV2/element_shape:output:0)sequential_2/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sequential_2/rnn/TensorArrayV2?
Fsequential_2/rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2H
Fsequential_2/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape?
8sequential_2/rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential_2/rnn/transpose:y:0Osequential_2/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02:
8sequential_2/rnn/TensorArrayUnstack/TensorListFromTensor?
&sequential_2/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_2/rnn/strided_slice_2/stack?
(sequential_2/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_2/rnn/strided_slice_2/stack_1?
(sequential_2/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_2/rnn/strided_slice_2/stack_2?
 sequential_2/rnn/strided_slice_2StridedSlicesequential_2/rnn/transpose:y:0/sequential_2/rnn/strided_slice_2/stack:output:01sequential_2/rnn/strided_slice_2/stack_1:output:01sequential_2/rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2"
 sequential_2/rnn/strided_slice_2?
9sequential_2/rnn/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOpBsequential_2_rnn_peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02;
9sequential_2/rnn/peephole_lstm_cell/MatMul/ReadVariableOp?
*sequential_2/rnn/peephole_lstm_cell/MatMulMatMul)sequential_2/rnn/strided_slice_2:output:0Asequential_2/rnn/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_2/rnn/peephole_lstm_cell/MatMul?
;sequential_2/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOpDsequential_2_rnn_peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02=
;sequential_2/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp?
,sequential_2/rnn/peephole_lstm_cell/MatMul_1MatMulsequential_2/rnn/zeros:output:0Csequential_2/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,sequential_2/rnn/peephole_lstm_cell/MatMul_1?
'sequential_2/rnn/peephole_lstm_cell/addAddV24sequential_2/rnn/peephole_lstm_cell/MatMul:product:06sequential_2/rnn/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2)
'sequential_2/rnn/peephole_lstm_cell/add?
:sequential_2/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOpCsequential_2_rnn_peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:sequential_2/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp?
+sequential_2/rnn/peephole_lstm_cell/BiasAddBiasAdd+sequential_2/rnn/peephole_lstm_cell/add:z:0Bsequential_2/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+sequential_2/rnn/peephole_lstm_cell/BiasAdd?
)sequential_2/rnn/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_2/rnn/peephole_lstm_cell/Const?
3sequential_2/rnn/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3sequential_2/rnn/peephole_lstm_cell/split/split_dim?
)sequential_2/rnn/peephole_lstm_cell/splitSplit<sequential_2/rnn/peephole_lstm_cell/split/split_dim:output:04sequential_2/rnn/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2+
)sequential_2/rnn/peephole_lstm_cell/split?
2sequential_2/rnn/peephole_lstm_cell/ReadVariableOpReadVariableOp;sequential_2_rnn_peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype024
2sequential_2/rnn/peephole_lstm_cell/ReadVariableOp?
'sequential_2/rnn/peephole_lstm_cell/mulMul:sequential_2/rnn/peephole_lstm_cell/ReadVariableOp:value:0!sequential_2/rnn/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2)
'sequential_2/rnn/peephole_lstm_cell/mul?
)sequential_2/rnn/peephole_lstm_cell/add_1AddV22sequential_2/rnn/peephole_lstm_cell/split:output:0+sequential_2/rnn/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2+
)sequential_2/rnn/peephole_lstm_cell/add_1?
+sequential_2/rnn/peephole_lstm_cell/SigmoidSigmoid-sequential_2/rnn/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2-
+sequential_2/rnn/peephole_lstm_cell/Sigmoid?
4sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp=sequential_2_rnn_peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype026
4sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_1?
)sequential_2/rnn/peephole_lstm_cell/mul_1Mul<sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_1:value:0!sequential_2/rnn/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2+
)sequential_2/rnn/peephole_lstm_cell/mul_1?
)sequential_2/rnn/peephole_lstm_cell/add_2AddV22sequential_2/rnn/peephole_lstm_cell/split:output:1-sequential_2/rnn/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2+
)sequential_2/rnn/peephole_lstm_cell/add_2?
-sequential_2/rnn/peephole_lstm_cell/Sigmoid_1Sigmoid-sequential_2/rnn/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2/
-sequential_2/rnn/peephole_lstm_cell/Sigmoid_1?
)sequential_2/rnn/peephole_lstm_cell/mul_2Mul1sequential_2/rnn/peephole_lstm_cell/Sigmoid_1:y:0!sequential_2/rnn/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2+
)sequential_2/rnn/peephole_lstm_cell/mul_2?
(sequential_2/rnn/peephole_lstm_cell/TanhTanh2sequential_2/rnn/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2*
(sequential_2/rnn/peephole_lstm_cell/Tanh?
)sequential_2/rnn/peephole_lstm_cell/mul_3Mul/sequential_2/rnn/peephole_lstm_cell/Sigmoid:y:0,sequential_2/rnn/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2+
)sequential_2/rnn/peephole_lstm_cell/mul_3?
)sequential_2/rnn/peephole_lstm_cell/add_3AddV2-sequential_2/rnn/peephole_lstm_cell/mul_2:z:0-sequential_2/rnn/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2+
)sequential_2/rnn/peephole_lstm_cell/add_3?
4sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp=sequential_2_rnn_peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype026
4sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_2?
)sequential_2/rnn/peephole_lstm_cell/mul_4Mul<sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_2:value:0-sequential_2/rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2+
)sequential_2/rnn/peephole_lstm_cell/mul_4?
)sequential_2/rnn/peephole_lstm_cell/add_4AddV22sequential_2/rnn/peephole_lstm_cell/split:output:3-sequential_2/rnn/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2+
)sequential_2/rnn/peephole_lstm_cell/add_4?
-sequential_2/rnn/peephole_lstm_cell/Sigmoid_2Sigmoid-sequential_2/rnn/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2/
-sequential_2/rnn/peephole_lstm_cell/Sigmoid_2?
*sequential_2/rnn/peephole_lstm_cell/Tanh_1Tanh-sequential_2/rnn/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2,
*sequential_2/rnn/peephole_lstm_cell/Tanh_1?
)sequential_2/rnn/peephole_lstm_cell/mul_5Mul1sequential_2/rnn/peephole_lstm_cell/Sigmoid_2:y:0.sequential_2/rnn/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2+
)sequential_2/rnn/peephole_lstm_cell/mul_5?
.sequential_2/rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    20
.sequential_2/rnn/TensorArrayV2_1/element_shape?
 sequential_2/rnn/TensorArrayV2_1TensorListReserve7sequential_2/rnn/TensorArrayV2_1/element_shape:output:0)sequential_2/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 sequential_2/rnn/TensorArrayV2_1p
sequential_2/rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_2/rnn/time?
)sequential_2/rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)sequential_2/rnn/while/maximum_iterations?
#sequential_2/rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_2/rnn/while/loop_counter?	
sequential_2/rnn/whileWhile,sequential_2/rnn/while/loop_counter:output:02sequential_2/rnn/while/maximum_iterations:output:0sequential_2/rnn/time:output:0)sequential_2/rnn/TensorArrayV2_1:handle:0sequential_2/rnn/zeros:output:0!sequential_2/rnn/zeros_1:output:0)sequential_2/rnn/strided_slice_1:output:0Hsequential_2/rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Bsequential_2_rnn_peephole_lstm_cell_matmul_readvariableop_resourceDsequential_2_rnn_peephole_lstm_cell_matmul_1_readvariableop_resourceCsequential_2_rnn_peephole_lstm_cell_biasadd_readvariableop_resource;sequential_2_rnn_peephole_lstm_cell_readvariableop_resource=sequential_2_rnn_peephole_lstm_cell_readvariableop_1_resource=sequential_2_rnn_peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
*-
body%R#
!sequential_2_rnn_while_body_27650*-
cond%R#
!sequential_2_rnn_while_cond_27649*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
sequential_2/rnn/while?
Asequential_2/rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2C
Asequential_2/rnn/TensorArrayV2Stack/TensorListStack/element_shape?
3sequential_2/rnn/TensorArrayV2Stack/TensorListStackTensorListStacksequential_2/rnn/while:output:3Jsequential_2/rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:2????????? *
element_dtype025
3sequential_2/rnn/TensorArrayV2Stack/TensorListStack?
&sequential_2/rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2(
&sequential_2/rnn/strided_slice_3/stack?
(sequential_2/rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_2/rnn/strided_slice_3/stack_1?
(sequential_2/rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_2/rnn/strided_slice_3/stack_2?
 sequential_2/rnn/strided_slice_3StridedSlice<sequential_2/rnn/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_2/rnn/strided_slice_3/stack:output:01sequential_2/rnn/strided_slice_3/stack_1:output:01sequential_2/rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2"
 sequential_2/rnn/strided_slice_3?
!sequential_2/rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!sequential_2/rnn/transpose_1/perm?
sequential_2/rnn/transpose_1	Transpose<sequential_2/rnn/TensorArrayV2Stack/TensorListStack:tensor:0*sequential_2/rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2 2
sequential_2/rnn/transpose_1?
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_2/dense_3/MatMul/ReadVariableOp?
sequential_2/dense_3/MatMulMatMul)sequential_2/rnn/strided_slice_3:output:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_3/MatMul?
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_2/dense_3/BiasAdd/ReadVariableOp?
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_3/BiasAdd?
sequential_2/dense_3/SigmoidSigmoid%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_2/dense_3/Sigmoid?
IdentityIdentity sequential_2/dense_3/Sigmoid:y:0,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp*^sequential_2/embedding_2/embedding_lookup;^sequential_2/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp:^sequential_2/rnn/peephole_lstm_cell/MatMul/ReadVariableOp<^sequential_2/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp3^sequential_2/rnn/peephole_lstm_cell/ReadVariableOp5^sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_15^sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_2^sequential_2/rnn/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2V
)sequential_2/embedding_2/embedding_lookup)sequential_2/embedding_2/embedding_lookup2x
:sequential_2/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp:sequential_2/rnn/peephole_lstm_cell/BiasAdd/ReadVariableOp2v
9sequential_2/rnn/peephole_lstm_cell/MatMul/ReadVariableOp9sequential_2/rnn/peephole_lstm_cell/MatMul/ReadVariableOp2z
;sequential_2/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp;sequential_2/rnn/peephole_lstm_cell/MatMul_1/ReadVariableOp2h
2sequential_2/rnn/peephole_lstm_cell/ReadVariableOp2sequential_2/rnn/peephole_lstm_cell/ReadVariableOp2l
4sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_14sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_12l
4sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_24sequential_2/rnn/peephole_lstm_cell/ReadVariableOp_220
sequential_2/rnn/whilesequential_2/rnn/while:Z V
'
_output_shapes
:?????????2
+
_user_specified_nameembedding_2_input
?
?
2__inference_peephole_lstm_cell_layer_call_fn_30685

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_279102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*d
_input_shapesS
Q:????????? :????????? :????????? ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?=
?
__inference__traced_save_30803
file_prefix5
1savev2_embedding_2_embeddings_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop<
8savev2_rnn_peephole_lstm_cell_kernel_read_readvariableopF
Bsavev2_rnn_peephole_lstm_cell_recurrent_kernel_read_readvariableop:
6savev2_rnn_peephole_lstm_cell_bias_read_readvariableopQ
Msavev2_rnn_peephole_lstm_cell_input_gate_peephole_weights_read_readvariableopR
Nsavev2_rnn_peephole_lstm_cell_forget_gate_peephole_weights_read_readvariableopR
Nsavev2_rnn_peephole_lstm_cell_output_gate_peephole_weights_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopB
>savev2_sgd_embedding_2_embeddings_momentum_read_readvariableop:
6savev2_sgd_dense_3_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_3_bias_momentum_read_readvariableopI
Esavev2_sgd_rnn_peephole_lstm_cell_kernel_momentum_read_readvariableopS
Osavev2_sgd_rnn_peephole_lstm_cell_recurrent_kernel_momentum_read_readvariableopG
Csavev2_sgd_rnn_peephole_lstm_cell_bias_momentum_read_readvariableop^
Zsavev2_sgd_rnn_peephole_lstm_cell_input_gate_peephole_weights_momentum_read_readvariableop_
[savev2_sgd_rnn_peephole_lstm_cell_forget_gate_peephole_weights_momentum_read_readvariableop_
[savev2_sgd_rnn_peephole_lstm_cell_output_gate_peephole_weights_momentum_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_2_embeddings_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop8savev2_rnn_peephole_lstm_cell_kernel_read_readvariableopBsavev2_rnn_peephole_lstm_cell_recurrent_kernel_read_readvariableop6savev2_rnn_peephole_lstm_cell_bias_read_readvariableopMsavev2_rnn_peephole_lstm_cell_input_gate_peephole_weights_read_readvariableopNsavev2_rnn_peephole_lstm_cell_forget_gate_peephole_weights_read_readvariableopNsavev2_rnn_peephole_lstm_cell_output_gate_peephole_weights_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_sgd_embedding_2_embeddings_momentum_read_readvariableop6savev2_sgd_dense_3_kernel_momentum_read_readvariableop4savev2_sgd_dense_3_bias_momentum_read_readvariableopEsavev2_sgd_rnn_peephole_lstm_cell_kernel_momentum_read_readvariableopOsavev2_sgd_rnn_peephole_lstm_cell_recurrent_kernel_momentum_read_readvariableopCsavev2_sgd_rnn_peephole_lstm_cell_bias_momentum_read_readvariableopZsavev2_sgd_rnn_peephole_lstm_cell_input_gate_peephole_weights_momentum_read_readvariableop[savev2_sgd_rnn_peephole_lstm_cell_forget_gate_peephole_weights_momentum_read_readvariableop[savev2_sgd_rnn_peephole_lstm_cell_output_gate_peephole_weights_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	? : :: : : : :	 ?:	 ?:?: : : : : :	? : ::	 ?:	 ?:?: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	? :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	 ?:%	!

_output_shapes
:	 ?:!


_output_shapes	
:?: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	? :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	 ?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: 
?d
?
while_body_28876
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_peephole_lstm_cell_matmul_readvariableop_resource_0?
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0>
:while_peephole_lstm_cell_biasadd_readvariableop_resource_06
2while_peephole_lstm_cell_readvariableop_resource_08
4while_peephole_lstm_cell_readvariableop_1_resource_08
4while_peephole_lstm_cell_readvariableop_2_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_peephole_lstm_cell_matmul_readvariableop_resource=
9while_peephole_lstm_cell_matmul_1_readvariableop_resource<
8while_peephole_lstm_cell_biasadd_readvariableop_resource4
0while_peephole_lstm_cell_readvariableop_resource6
2while_peephole_lstm_cell_readvariableop_1_resource6
2while_peephole_lstm_cell_readvariableop_2_resource??/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?.while/peephole_lstm_cell/MatMul/ReadVariableOp?0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?'while/peephole_lstm_cell/ReadVariableOp?)while/peephole_lstm_cell/ReadVariableOp_1?)while/peephole_lstm_cell/ReadVariableOp_2?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOp?
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/peephole_lstm_cell/MatMul?
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/peephole_lstm_cell/MatMul_1?
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/peephole_lstm_cell/add?
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/peephole_lstm_cell/BiasAdd?
while/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
while/peephole_lstm_cell/Const?
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim?
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2 
while/peephole_lstm_cell/split?
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp?
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/mul?
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_1?
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 while/peephole_lstm_cell/Sigmoid?
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1?
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_1?
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_2?
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_1?
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_2?
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/Tanh?
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_3?
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_3?
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2?
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_4?
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_4?
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_2?
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2!
while/peephole_lstm_cell/Tanh_1?
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?d
?
while_body_30018
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_peephole_lstm_cell_matmul_readvariableop_resource_0?
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0>
:while_peephole_lstm_cell_biasadd_readvariableop_resource_06
2while_peephole_lstm_cell_readvariableop_resource_08
4while_peephole_lstm_cell_readvariableop_1_resource_08
4while_peephole_lstm_cell_readvariableop_2_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_peephole_lstm_cell_matmul_readvariableop_resource=
9while_peephole_lstm_cell_matmul_1_readvariableop_resource<
8while_peephole_lstm_cell_biasadd_readvariableop_resource4
0while_peephole_lstm_cell_readvariableop_resource6
2while_peephole_lstm_cell_readvariableop_1_resource6
2while_peephole_lstm_cell_readvariableop_2_resource??/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?.while/peephole_lstm_cell/MatMul/ReadVariableOp?0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?'while/peephole_lstm_cell/ReadVariableOp?)while/peephole_lstm_cell/ReadVariableOp_1?)while/peephole_lstm_cell/ReadVariableOp_2?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOp?
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/peephole_lstm_cell/MatMul?
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/peephole_lstm_cell/MatMul_1?
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/peephole_lstm_cell/add?
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/peephole_lstm_cell/BiasAdd?
while/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
while/peephole_lstm_cell/Const?
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim?
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2 
while/peephole_lstm_cell/split?
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp?
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/mul?
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_1?
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 while/peephole_lstm_cell/Sigmoid?
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1?
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_1?
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_2?
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_1?
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_2?
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/Tanh?
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_3?
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_3?
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2?
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_4?
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_4?
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_2?
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2!
while/peephole_lstm_cell/Tanh_1?
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
#__inference_rnn_layer_call_fn_30154

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_289782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????2 ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29158

inputs
embedding_2_29135
	rnn_29139
	rnn_29141
	rnn_29143
	rnn_29145
	rnn_29147
	rnn_29149
dense_3_29152
dense_3_29154
identity??dense_3/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?rnn/StatefulPartitionedCall?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_2_29135*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_285652%
#embedding_2/StatefulPartitionedCall?
#spatial_dropout1d_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_286032%
#spatial_dropout1d_2/PartitionedCall?
rnn/StatefulPartitionedCallStatefulPartitionedCall,spatial_dropout1d_2/PartitionedCall:output:0	rnn_29139	rnn_29141	rnn_29143	rnn_29145	rnn_29147	rnn_29149*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_289782
rnn/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_3_29152dense_3_29154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_290372!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?'
?
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_27955

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
readvariableop_resource
readvariableop_1_resource
readvariableop_2_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
splitt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0states_1*
T0*'
_output_shapes
:????????? 2
mulb
add_1AddV2split:output:0mul:z:0*
T0*'
_output_shapes
:????????? 2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1k
mul_1MulReadVariableOp_1:value:0states_1*
T0*'
_output_shapes
:????????? 2
mul_1d
add_2AddV2split:output:1	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_2^
	Sigmoid_1Sigmoid	add_2:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1`
mul_2MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mul_2V
TanhTanhsplit:output:2*
T0*'
_output_shapes
:????????? 2
Tanh^
mul_3MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:????????? 2
mul_3_
add_3AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:????????? 2
add_3z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2l
mul_4MulReadVariableOp_2:value:0	add_3:z:0*
T0*'
_output_shapes
:????????? 2
mul_4d
add_4AddV2split:output:3	mul_4:z:0*
T0*'
_output_shapes
:????????? 2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:????????? 2
Tanh_1b
mul_5MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
mul_5?
IdentityIdentity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity	mul_5:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*d
_input_shapesS
Q:????????? :????????? :????????? ::::::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_2:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
?	
?
while_cond_30415
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_30415___redundant_placeholder03
/while_while_cond_30415___redundant_placeholder13
/while_while_cond_30415___redundant_placeholder23
/while_while_cond_30415___redundant_placeholder33
/while_while_cond_30415___redundant_placeholder43
/while_while_cond_30415___redundant_placeholder53
/while_while_cond_30415___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
#__inference_rnn_layer_call_fn_30552
inputs_0
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_285362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????????????? ::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?	
?
while_cond_29835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_29835___redundant_placeholder03
/while_while_cond_29835___redundant_placeholder13
/while_while_cond_29835___redundant_placeholder23
/while_while_cond_29835___redundant_placeholder33
/while_while_cond_29835___redundant_placeholder43
/while_while_cond_29835___redundant_placeholder53
/while_while_cond_29835___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
l
3__inference_spatial_dropout1d_2_layer_call_fn_29714

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_278122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
while_cond_28300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_28300___redundant_placeholder03
/while_while_cond_28300___redundant_placeholder13
/while_while_cond_28300___redundant_placeholder23
/while_while_cond_28300___redundant_placeholder33
/while_while_cond_28300___redundant_placeholder43
/while_while_cond_28300___redundant_placeholder53
/while_while_cond_28300___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?	
?
while_cond_28875
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_28875___redundant_placeholder03
/while_while_cond_28875___redundant_placeholder13
/while_while_cond_28875___redundant_placeholder23
/while_while_cond_28875___redundant_placeholder33
/while_while_cond_28875___redundant_placeholder43
/while_while_cond_28875___redundant_placeholder53
/while_while_cond_28875___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?

?
rnn_while_cond_29313$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_29313___redundant_placeholder0;
7rnn_while_rnn_while_cond_29313___redundant_placeholder1;
7rnn_while_rnn_while_cond_29313___redundant_placeholder2;
7rnn_while_rnn_while_cond_29313___redundant_placeholder3;
7rnn_while_rnn_while_cond_29313___redundant_placeholder4;
7rnn_while_rnn_while_cond_29313___redundant_placeholder5;
7rnn_while_rnn_while_cond_29313___redundant_placeholder6
rnn_while_identity
?
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: 2
rnn/while/Lessi
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn/while/Identity"1
rnn_while_identityrnn/while/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?

?
rnn_while_cond_29509$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_29509___redundant_placeholder0;
7rnn_while_rnn_while_cond_29509___redundant_placeholder1;
7rnn_while_rnn_while_cond_29509___redundant_placeholder2;
7rnn_while_rnn_while_cond_29509___redundant_placeholder3;
7rnn_while_rnn_while_cond_29509___redundant_placeholder4;
7rnn_while_rnn_while_cond_29509___redundant_placeholder5;
7rnn_while_rnn_while_cond_29509___redundant_placeholder6
rnn_while_identity
?
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: 2
rnn/while/Lessi
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn/while/Identity"1
rnn_while_identityrnn/while/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?l
?
rnn_while_body_29510$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0A
=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0C
?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0B
>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0:
6rnn_while_peephole_lstm_cell_readvariableop_resource_0<
8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0<
8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor?
;rnn_while_peephole_lstm_cell_matmul_readvariableop_resourceA
=rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource@
<rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource8
4rnn_while_peephole_lstm_cell_readvariableop_resource:
6rnn_while_peephole_lstm_cell_readvariableop_1_resource:
6rnn_while_peephole_lstm_cell_readvariableop_2_resource??3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp?4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp?+rnn/while/peephole_lstm_cell/ReadVariableOp?-rnn/while/peephole_lstm_cell/ReadVariableOp_1?-rnn/while/peephole_lstm_cell/ReadVariableOp_2?
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2=
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02/
-rnn/while/TensorArrayV2Read/TensorListGetItem?
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype024
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp?
#rnn/while/peephole_lstm_cell/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:0:rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#rnn/while/peephole_lstm_cell/MatMul?
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype026
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp?
%rnn/while/peephole_lstm_cell/MatMul_1MatMulrnn_while_placeholder_2<rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%rnn/while/peephole_lstm_cell/MatMul_1?
 rnn/while/peephole_lstm_cell/addAddV2-rnn/while/peephole_lstm_cell/MatMul:product:0/rnn/while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2"
 rnn/while/peephole_lstm_cell/add?
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype025
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?
$rnn/while/peephole_lstm_cell/BiasAddBiasAdd$rnn/while/peephole_lstm_cell/add:z:0;rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2&
$rnn/while/peephole_lstm_cell/BiasAdd?
"rnn/while/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"rnn/while/peephole_lstm_cell/Const?
,rnn/while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,rnn/while/peephole_lstm_cell/split/split_dim?
"rnn/while/peephole_lstm_cell/splitSplit5rnn/while/peephole_lstm_cell/split/split_dim:output:0-rnn/while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2$
"rnn/while/peephole_lstm_cell/split?
+rnn/while/peephole_lstm_cell/ReadVariableOpReadVariableOp6rnn_while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02-
+rnn/while/peephole_lstm_cell/ReadVariableOp?
 rnn/while/peephole_lstm_cell/mulMul3rnn/while/peephole_lstm_cell/ReadVariableOp:value:0rnn_while_placeholder_3*
T0*'
_output_shapes
:????????? 2"
 rnn/while/peephole_lstm_cell/mul?
"rnn/while/peephole_lstm_cell/add_1AddV2+rnn/while/peephole_lstm_cell/split:output:0$rnn/while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/add_1?
$rnn/while/peephole_lstm_cell/SigmoidSigmoid&rnn/while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2&
$rnn/while/peephole_lstm_cell/Sigmoid?
-rnn/while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02/
-rnn/while/peephole_lstm_cell/ReadVariableOp_1?
"rnn/while/peephole_lstm_cell/mul_1Mul5rnn/while/peephole_lstm_cell/ReadVariableOp_1:value:0rnn_while_placeholder_3*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_1?
"rnn/while/peephole_lstm_cell/add_2AddV2+rnn/while/peephole_lstm_cell/split:output:1&rnn/while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/add_2?
&rnn/while/peephole_lstm_cell/Sigmoid_1Sigmoid&rnn/while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2(
&rnn/while/peephole_lstm_cell/Sigmoid_1?
"rnn/while/peephole_lstm_cell/mul_2Mul*rnn/while/peephole_lstm_cell/Sigmoid_1:y:0rnn_while_placeholder_3*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_2?
!rnn/while/peephole_lstm_cell/TanhTanh+rnn/while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2#
!rnn/while/peephole_lstm_cell/Tanh?
"rnn/while/peephole_lstm_cell/mul_3Mul(rnn/while/peephole_lstm_cell/Sigmoid:y:0%rnn/while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_3?
"rnn/while/peephole_lstm_cell/add_3AddV2&rnn/while/peephole_lstm_cell/mul_2:z:0&rnn/while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/add_3?
-rnn/while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02/
-rnn/while/peephole_lstm_cell/ReadVariableOp_2?
"rnn/while/peephole_lstm_cell/mul_4Mul5rnn/while/peephole_lstm_cell/ReadVariableOp_2:value:0&rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_4?
"rnn/while/peephole_lstm_cell/add_4AddV2+rnn/while/peephole_lstm_cell/split:output:3&rnn/while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/add_4?
&rnn/while/peephole_lstm_cell/Sigmoid_2Sigmoid&rnn/while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2(
&rnn/while/peephole_lstm_cell/Sigmoid_2?
#rnn/while/peephole_lstm_cell/Tanh_1Tanh&rnn/while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2%
#rnn/while/peephole_lstm_cell/Tanh_1?
"rnn/while/peephole_lstm_cell/mul_5Mul*rnn/while/peephole_lstm_cell/Sigmoid_2:y:0'rnn/while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2$
"rnn/while/peephole_lstm_cell/mul_5?
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder&rnn/while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype020
.rnn/while/TensorArrayV2Write/TensorListSetItemd
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add/yy
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn/while/addh
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add_1/y?
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn/while/add_1?
rnn/while/IdentityIdentityrnn/while/add_1:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity?
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations4^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_1?
rnn/while/Identity_2Identityrnn/while/add:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_2?
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
rnn/while/Identity_3?
rnn/while/Identity_4Identity&rnn/while/peephole_lstm_cell/mul_5:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
rnn/while/Identity_4?
rnn/while/Identity_5Identity&rnn/while/peephole_lstm_cell/add_3:z:04^rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3^rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp5^rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp,^rnn/while/peephole_lstm_cell/ReadVariableOp.^rnn/while/peephole_lstm_cell/ReadVariableOp_1.^rnn/while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
rnn/while/Identity_5"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"~
<rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource>rnn_while_peephole_lstm_cell_biasadd_readvariableop_resource_0"?
=rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource?rnn_while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"|
;rnn_while_peephole_lstm_cell_matmul_readvariableop_resource=rnn_while_peephole_lstm_cell_matmul_readvariableop_resource_0"r
6rnn_while_peephole_lstm_cell_readvariableop_1_resource8rnn_while_peephole_lstm_cell_readvariableop_1_resource_0"r
6rnn_while_peephole_lstm_cell_readvariableop_2_resource8rnn_while_peephole_lstm_cell_readvariableop_2_resource_0"n
4rnn_while_peephole_lstm_cell_readvariableop_resource6rnn_while_peephole_lstm_cell_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"?
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2j
3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp3rnn/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2h
2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2rnn/while/peephole_lstm_cell/MatMul/ReadVariableOp2l
4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp4rnn/while/peephole_lstm_cell/MatMul_1/ReadVariableOp2Z
+rnn/while/peephole_lstm_cell/ReadVariableOp+rnn/while/peephole_lstm_cell/ReadVariableOp2^
-rnn/while/peephole_lstm_cell/ReadVariableOp_1-rnn/while/peephole_lstm_cell/ReadVariableOp_12^
-rnn/while/peephole_lstm_cell/ReadVariableOp_2-rnn/while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?d
?
while_body_30234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
9while_peephole_lstm_cell_matmul_readvariableop_resource_0?
;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0>
:while_peephole_lstm_cell_biasadd_readvariableop_resource_06
2while_peephole_lstm_cell_readvariableop_resource_08
4while_peephole_lstm_cell_readvariableop_1_resource_08
4while_peephole_lstm_cell_readvariableop_2_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
7while_peephole_lstm_cell_matmul_readvariableop_resource=
9while_peephole_lstm_cell_matmul_1_readvariableop_resource<
8while_peephole_lstm_cell_biasadd_readvariableop_resource4
0while_peephole_lstm_cell_readvariableop_resource6
2while_peephole_lstm_cell_readvariableop_1_resource6
2while_peephole_lstm_cell_readvariableop_2_resource??/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?.while/peephole_lstm_cell/MatMul/ReadVariableOp?0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?'while/peephole_lstm_cell/ReadVariableOp?)while/peephole_lstm_cell/ReadVariableOp_1?)while/peephole_lstm_cell/ReadVariableOp_2?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
.while/peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp9while_peephole_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype020
.while/peephole_lstm_cell/MatMul/ReadVariableOp?
while/peephole_lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:06while/peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
while/peephole_lstm_cell/MatMul?
0while/peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype022
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp?
!while/peephole_lstm_cell/MatMul_1MatMulwhile_placeholder_28while/peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!while/peephole_lstm_cell/MatMul_1?
while/peephole_lstm_cell/addAddV2)while/peephole_lstm_cell/MatMul:product:0+while/peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/peephole_lstm_cell/add?
/while/peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:while_peephole_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype021
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp?
 while/peephole_lstm_cell/BiasAddBiasAdd while/peephole_lstm_cell/add:z:07while/peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 while/peephole_lstm_cell/BiasAdd?
while/peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
while/peephole_lstm_cell/Const?
(while/peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(while/peephole_lstm_cell/split/split_dim?
while/peephole_lstm_cell/splitSplit1while/peephole_lstm_cell/split/split_dim:output:0)while/peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2 
while/peephole_lstm_cell/split?
'while/peephole_lstm_cell/ReadVariableOpReadVariableOp2while_peephole_lstm_cell_readvariableop_resource_0*
_output_shapes
: *
dtype02)
'while/peephole_lstm_cell/ReadVariableOp?
while/peephole_lstm_cell/mulMul/while/peephole_lstm_cell/ReadVariableOp:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/mul?
while/peephole_lstm_cell/add_1AddV2'while/peephole_lstm_cell/split:output:0 while/peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_1?
 while/peephole_lstm_cell/SigmoidSigmoid"while/peephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 while/peephole_lstm_cell/Sigmoid?
)while/peephole_lstm_cell/ReadVariableOp_1ReadVariableOp4while_peephole_lstm_cell_readvariableop_1_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_1?
while/peephole_lstm_cell/mul_1Mul1while/peephole_lstm_cell/ReadVariableOp_1:value:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_1?
while/peephole_lstm_cell/add_2AddV2'while/peephole_lstm_cell/split:output:1"while/peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_2?
"while/peephole_lstm_cell/Sigmoid_1Sigmoid"while/peephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_1?
while/peephole_lstm_cell/mul_2Mul&while/peephole_lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_2?
while/peephole_lstm_cell/TanhTanh'while/peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
while/peephole_lstm_cell/Tanh?
while/peephole_lstm_cell/mul_3Mul$while/peephole_lstm_cell/Sigmoid:y:0!while/peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_3?
while/peephole_lstm_cell/add_3AddV2"while/peephole_lstm_cell/mul_2:z:0"while/peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_3?
)while/peephole_lstm_cell/ReadVariableOp_2ReadVariableOp4while_peephole_lstm_cell_readvariableop_2_resource_0*
_output_shapes
: *
dtype02+
)while/peephole_lstm_cell/ReadVariableOp_2?
while/peephole_lstm_cell/mul_4Mul1while/peephole_lstm_cell/ReadVariableOp_2:value:0"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_4?
while/peephole_lstm_cell/add_4AddV2'while/peephole_lstm_cell/split:output:3"while/peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/add_4?
"while/peephole_lstm_cell/Sigmoid_2Sigmoid"while/peephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2$
"while/peephole_lstm_cell/Sigmoid_2?
while/peephole_lstm_cell/Tanh_1Tanh"while/peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2!
while/peephole_lstm_cell/Tanh_1?
while/peephole_lstm_cell/mul_5Mul&while/peephole_lstm_cell/Sigmoid_2:y:0#while/peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2 
while/peephole_lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/peephole_lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations0^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity"while/peephole_lstm_cell/mul_5:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity"while/peephole_lstm_cell/add_3:z:00^while/peephole_lstm_cell/BiasAdd/ReadVariableOp/^while/peephole_lstm_cell/MatMul/ReadVariableOp1^while/peephole_lstm_cell/MatMul_1/ReadVariableOp(^while/peephole_lstm_cell/ReadVariableOp*^while/peephole_lstm_cell/ReadVariableOp_1*^while/peephole_lstm_cell/ReadVariableOp_2*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"v
8while_peephole_lstm_cell_biasadd_readvariableop_resource:while_peephole_lstm_cell_biasadd_readvariableop_resource_0"x
9while_peephole_lstm_cell_matmul_1_readvariableop_resource;while_peephole_lstm_cell_matmul_1_readvariableop_resource_0"t
7while_peephole_lstm_cell_matmul_readvariableop_resource9while_peephole_lstm_cell_matmul_readvariableop_resource_0"j
2while_peephole_lstm_cell_readvariableop_1_resource4while_peephole_lstm_cell_readvariableop_1_resource_0"j
2while_peephole_lstm_cell_readvariableop_2_resource4while_peephole_lstm_cell_readvariableop_2_resource_0"f
0while_peephole_lstm_cell_readvariableop_resource2while_peephole_lstm_cell_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2b
/while/peephole_lstm_cell/BiasAdd/ReadVariableOp/while/peephole_lstm_cell/BiasAdd/ReadVariableOp2`
.while/peephole_lstm_cell/MatMul/ReadVariableOp.while/peephole_lstm_cell/MatMul/ReadVariableOp2d
0while/peephole_lstm_cell/MatMul_1/ReadVariableOp0while/peephole_lstm_cell/MatMul_1/ReadVariableOp2R
'while/peephole_lstm_cell/ReadVariableOp'while/peephole_lstm_cell/ReadVariableOp2V
)while/peephole_lstm_cell/ReadVariableOp_1)while/peephole_lstm_cell/ReadVariableOp_12V
)while/peephole_lstm_cell/ReadVariableOp_2)while/peephole_lstm_cell/ReadVariableOp_2: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
O
3__inference_spatial_dropout1d_2_layer_call_fn_29756

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_286032
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2 :S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_30563

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_29037

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?p
?
>__inference_rnn_layer_call_and_return_conditional_losses_30518
inputs_05
1peephole_lstm_cell_matmul_readvariableop_resource7
3peephole_lstm_cell_matmul_1_readvariableop_resource6
2peephole_lstm_cell_biasadd_readvariableop_resource.
*peephole_lstm_cell_readvariableop_resource0
,peephole_lstm_cell_readvariableop_1_resource0
,peephole_lstm_cell_readvariableop_2_resource
identity??)peephole_lstm_cell/BiasAdd/ReadVariableOp?(peephole_lstm_cell/MatMul/ReadVariableOp?*peephole_lstm_cell/MatMul_1/ReadVariableOp?!peephole_lstm_cell/ReadVariableOp?#peephole_lstm_cell/ReadVariableOp_1?#peephole_lstm_cell/ReadVariableOp_2?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_2?
(peephole_lstm_cell/MatMul/ReadVariableOpReadVariableOp1peephole_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02*
(peephole_lstm_cell/MatMul/ReadVariableOp?
peephole_lstm_cell/MatMulMatMulstrided_slice_2:output:00peephole_lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul?
*peephole_lstm_cell/MatMul_1/ReadVariableOpReadVariableOp3peephole_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02,
*peephole_lstm_cell/MatMul_1/ReadVariableOp?
peephole_lstm_cell/MatMul_1MatMulzeros:output:02peephole_lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/MatMul_1?
peephole_lstm_cell/addAddV2#peephole_lstm_cell/MatMul:product:0%peephole_lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/add?
)peephole_lstm_cell/BiasAdd/ReadVariableOpReadVariableOp2peephole_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)peephole_lstm_cell/BiasAdd/ReadVariableOp?
peephole_lstm_cell/BiasAddBiasAddpeephole_lstm_cell/add:z:01peephole_lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
peephole_lstm_cell/BiasAddv
peephole_lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
peephole_lstm_cell/Const?
"peephole_lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"peephole_lstm_cell/split/split_dim?
peephole_lstm_cell/splitSplit+peephole_lstm_cell/split/split_dim:output:0#peephole_lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
peephole_lstm_cell/split?
!peephole_lstm_cell/ReadVariableOpReadVariableOp*peephole_lstm_cell_readvariableop_resource*
_output_shapes
: *
dtype02#
!peephole_lstm_cell/ReadVariableOp?
peephole_lstm_cell/mulMul)peephole_lstm_cell/ReadVariableOp:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul?
peephole_lstm_cell/add_1AddV2!peephole_lstm_cell/split:output:0peephole_lstm_cell/mul:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_1?
peephole_lstm_cell/SigmoidSigmoidpeephole_lstm_cell/add_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid?
#peephole_lstm_cell/ReadVariableOp_1ReadVariableOp,peephole_lstm_cell_readvariableop_1_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_1?
peephole_lstm_cell/mul_1Mul+peephole_lstm_cell/ReadVariableOp_1:value:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_1?
peephole_lstm_cell/add_2AddV2!peephole_lstm_cell/split:output:1peephole_lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_2?
peephole_lstm_cell/Sigmoid_1Sigmoidpeephole_lstm_cell/add_2:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_1?
peephole_lstm_cell/mul_2Mul peephole_lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_2?
peephole_lstm_cell/TanhTanh!peephole_lstm_cell/split:output:2*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh?
peephole_lstm_cell/mul_3Mulpeephole_lstm_cell/Sigmoid:y:0peephole_lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_3?
peephole_lstm_cell/add_3AddV2peephole_lstm_cell/mul_2:z:0peephole_lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_3?
#peephole_lstm_cell/ReadVariableOp_2ReadVariableOp,peephole_lstm_cell_readvariableop_2_resource*
_output_shapes
: *
dtype02%
#peephole_lstm_cell/ReadVariableOp_2?
peephole_lstm_cell/mul_4Mul+peephole_lstm_cell/ReadVariableOp_2:value:0peephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_4?
peephole_lstm_cell/add_4AddV2!peephole_lstm_cell/split:output:3peephole_lstm_cell/mul_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/add_4?
peephole_lstm_cell/Sigmoid_2Sigmoidpeephole_lstm_cell/add_4:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Sigmoid_2?
peephole_lstm_cell/Tanh_1Tanhpeephole_lstm_cell/add_3:z:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/Tanh_1?
peephole_lstm_cell/mul_5Mul peephole_lstm_cell/Sigmoid_2:y:0peephole_lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:????????? 2
peephole_lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:01peephole_lstm_cell_matmul_readvariableop_resource3peephole_lstm_cell_matmul_1_readvariableop_resource2peephole_lstm_cell_biasadd_readvariableop_resource*peephole_lstm_cell_readvariableop_resource,peephole_lstm_cell_readvariableop_1_resource,peephole_lstm_cell_readvariableop_2_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_30416*
condR
while_cond_30415*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1?
IdentityIdentitystrided_slice_3:output:0*^peephole_lstm_cell/BiasAdd/ReadVariableOp)^peephole_lstm_cell/MatMul/ReadVariableOp+^peephole_lstm_cell/MatMul_1/ReadVariableOp"^peephole_lstm_cell/ReadVariableOp$^peephole_lstm_cell/ReadVariableOp_1$^peephole_lstm_cell/ReadVariableOp_2^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????????????? ::::::2V
)peephole_lstm_cell/BiasAdd/ReadVariableOp)peephole_lstm_cell/BiasAdd/ReadVariableOp2T
(peephole_lstm_cell/MatMul/ReadVariableOp(peephole_lstm_cell/MatMul/ReadVariableOp2X
*peephole_lstm_cell/MatMul_1/ReadVariableOp*peephole_lstm_cell/MatMul_1/ReadVariableOp2F
!peephole_lstm_cell/ReadVariableOp!peephole_lstm_cell/ReadVariableOp2J
#peephole_lstm_cell/ReadVariableOp_1#peephole_lstm_cell/ReadVariableOp_12J
#peephole_lstm_cell/ReadVariableOp_2#peephole_lstm_cell/ReadVariableOp_22
whilewhile:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?
O
3__inference_spatial_dropout1d_2_layer_call_fn_29719

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_278222
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?*
?
while_body_28456
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 while_peephole_lstm_cell_28480_0$
 while_peephole_lstm_cell_28482_0$
 while_peephole_lstm_cell_28484_0$
 while_peephole_lstm_cell_28486_0$
 while_peephole_lstm_cell_28488_0$
 while_peephole_lstm_cell_28490_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
while_peephole_lstm_cell_28480"
while_peephole_lstm_cell_28482"
while_peephole_lstm_cell_28484"
while_peephole_lstm_cell_28486"
while_peephole_lstm_cell_28488"
while_peephole_lstm_cell_28490??0while/peephole_lstm_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
0while/peephole_lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3 while_peephole_lstm_cell_28480_0 while_peephole_lstm_cell_28482_0 while_peephole_lstm_cell_28484_0 while_peephole_lstm_cell_28486_0 while_peephole_lstm_cell_28488_0 while_peephole_lstm_cell_28490_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_2795522
0while/peephole_lstm_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder9while/peephole_lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations1^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:01^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity9while/peephole_lstm_cell/StatefulPartitionedCall:output:11^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity9while/peephole_lstm_cell/StatefulPartitionedCall:output:21^while/peephole_lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"B
while_peephole_lstm_cell_28480 while_peephole_lstm_cell_28480_0"B
while_peephole_lstm_cell_28482 while_peephole_lstm_cell_28482_0"B
while_peephole_lstm_cell_28484 while_peephole_lstm_cell_28484_0"B
while_peephole_lstm_cell_28486 while_peephole_lstm_cell_28486_0"B
while_peephole_lstm_cell_28488 while_peephole_lstm_cell_28488_0"B
while_peephole_lstm_cell_28490 while_peephole_lstm_cell_28490_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :????????? :????????? : : ::::::2d
0while/peephole_lstm_cell/StatefulPartitionedCall0while/peephole_lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
!sequential_2_rnn_while_cond_27649>
:sequential_2_rnn_while_sequential_2_rnn_while_loop_counterD
@sequential_2_rnn_while_sequential_2_rnn_while_maximum_iterations&
"sequential_2_rnn_while_placeholder(
$sequential_2_rnn_while_placeholder_1(
$sequential_2_rnn_while_placeholder_2(
$sequential_2_rnn_while_placeholder_3@
<sequential_2_rnn_while_less_sequential_2_rnn_strided_slice_1U
Qsequential_2_rnn_while_sequential_2_rnn_while_cond_27649___redundant_placeholder0U
Qsequential_2_rnn_while_sequential_2_rnn_while_cond_27649___redundant_placeholder1U
Qsequential_2_rnn_while_sequential_2_rnn_while_cond_27649___redundant_placeholder2U
Qsequential_2_rnn_while_sequential_2_rnn_while_cond_27649___redundant_placeholder3U
Qsequential_2_rnn_while_sequential_2_rnn_while_cond_27649___redundant_placeholder4U
Qsequential_2_rnn_while_sequential_2_rnn_while_cond_27649___redundant_placeholder5U
Qsequential_2_rnn_while_sequential_2_rnn_while_cond_27649___redundant_placeholder6#
sequential_2_rnn_while_identity
?
sequential_2/rnn/while/LessLess"sequential_2_rnn_while_placeholder<sequential_2_rnn_while_less_sequential_2_rnn_strided_slice_1*
T0*
_output_shapes
: 2
sequential_2/rnn/while/Less?
sequential_2/rnn/while/IdentityIdentitysequential_2/rnn/while/Less:z:0*
T0
*
_output_shapes
: 2!
sequential_2/rnn/while/Identity"K
sequential_2_rnn_while_identity(sequential_2/rnn/while/Identity:output:0*_
_input_shapesN
L: : : : :????????? :????????? : :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29109

inputs
embedding_2_29086
	rnn_29090
	rnn_29092
	rnn_29094
	rnn_29096
	rnn_29098
	rnn_29100
dense_3_29103
dense_3_29105
identity??dense_3/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?rnn/StatefulPartitionedCall?+spatial_dropout1d_2/StatefulPartitionedCall?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_2_29086*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_285652%
#embedding_2/StatefulPartitionedCall?
+spatial_dropout1d_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_285982-
+spatial_dropout1d_2/StatefulPartitionedCall?
rnn/StatefulPartitionedCallStatefulPartitionedCall4spatial_dropout1d_2/StatefulPartitionedCall:output:0	rnn_29090	rnn_29092	rnn_29094	rnn_29096	rnn_29098	rnn_29100*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_287962
rnn/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_3_29103dense_3_29105*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_290372!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall^rnn/StatefulPartitionedCall,^spatial_dropout1d_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2Z
+spatial_dropout1d_2/StatefulPartitionedCall+spatial_dropout1d_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
2__inference_peephole_lstm_cell_layer_call_fn_30708

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_279552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*d
_input_shapesS
Q:????????? :????????? :????????? ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?F
?
>__inference_rnn_layer_call_and_return_conditional_losses_28381

inputs
peephole_lstm_cell_28282
peephole_lstm_cell_28284
peephole_lstm_cell_28286
peephole_lstm_cell_28288
peephole_lstm_cell_28290
peephole_lstm_cell_28292
identity??*peephole_lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_2?
*peephole_lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0peephole_lstm_cell_28282peephole_lstm_cell_28284peephole_lstm_cell_28286peephole_lstm_cell_28288peephole_lstm_cell_28290peephole_lstm_cell_28292*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_279102,
*peephole_lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0peephole_lstm_cell_28282peephole_lstm_cell_28284peephole_lstm_cell_28286peephole_lstm_cell_28288peephole_lstm_cell_28290peephole_lstm_cell_28292*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*R
_output_shapes@
>: : : : :????????? :????????? : : : : : : : : *(
_read_only_resource_inputs

	
*
bodyR
while_body_28301*
condR
while_cond_28300*Q
output_shapes@
>: : : : :????????? :????????? : : : : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
transpose_1?
IdentityIdentitystrided_slice_3:output:0+^peephole_lstm_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:?????????????????? ::::::2X
*peephole_lstm_cell/StatefulPartitionedCall*peephole_lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29054
embedding_2_input
embedding_2_28574
	rnn_29013
	rnn_29015
	rnn_29017
	rnn_29019
	rnn_29021
	rnn_29023
dense_3_29048
dense_3_29050
identity??dense_3/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?rnn/StatefulPartitionedCall?+spatial_dropout1d_2/StatefulPartitionedCall?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallembedding_2_inputembedding_2_28574*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_285652%
#embedding_2/StatefulPartitionedCall?
+spatial_dropout1d_2/StatefulPartitionedCallStatefulPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_285982-
+spatial_dropout1d_2/StatefulPartitionedCall?
rnn/StatefulPartitionedCallStatefulPartitionedCall4spatial_dropout1d_2/StatefulPartitionedCall:output:0	rnn_29013	rnn_29015	rnn_29017	rnn_29019	rnn_29021	rnn_29023*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_287962
rnn/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_3_29048dense_3_29050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_290372!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall^rnn/StatefulPartitionedCall,^spatial_dropout1d_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*J
_input_shapes9
7:?????????2:::::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2Z
+spatial_dropout1d_2/StatefulPartitionedCall+spatial_dropout1d_2/StatefulPartitionedCall:Z V
'
_output_shapes
:?????????2
+
_user_specified_nameembedding_2_input
?	
?
F__inference_embedding_2_layer_call_and_return_conditional_losses_29675

inputs
embedding_lookup_29669
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????22
Cast?
embedding_lookupResourceGatherembedding_lookup_29669Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/29669*+
_output_shapes
:?????????2 *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/29669*+
_output_shapes
:?????????2 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2 2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????2:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
embedding_2_input:
#serving_default_embedding_2_input:0?????????2;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?+
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
[__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses"?(
_tf_keras_sequential?({"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_2_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "input_dim": 306, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 50}}, {"class_name": "SpatialDropout1D", "config": {"name": "spatial_dropout1d_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_2_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "input_dim": 306, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 50}}, {"class_name": "SpatialDropout1D", "config": {"name": "spatial_dropout1d_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "root_mean_squared_logarithmic_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0010000000474974513, "decay": 9.999999974752427e-07, "momentum": 0.9900000095367432, "nesterov": true}}}}
?

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
^__call__
*_&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "input_dim": 306, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
`__call__
*a&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "SpatialDropout1D", "name": "spatial_dropout1d_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout1d_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
b__call__
*c&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?
{"class_name": "RNN", "name": "rnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "Addons>PeepholeLSTMCell", "config": {"name": "peephole_lstm_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 32]}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
 iter
	!decay
"learning_rate
#momentummomentumRmomentumSmomentumT$momentumU%momentumV&momentumW'momentumX(momentumY)momentumZ"
	optimizer
 "
trackable_list_wrapper
_
0
$1
%2
&3
'4
(5
)6
7
8"
trackable_list_wrapper
_
0
$1
%2
&3
'4
(5
)6
7
8"
trackable_list_wrapper
?
*layer_metrics

+layers
regularization_losses
,non_trainable_variables
	variables
-layer_regularization_losses
trainable_variables
.metrics
[__call__
\_default_save_signature
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
,
fserving_default"
signature_map
):'	? 2embedding_2/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
/layer_metrics

0layers
regularization_losses
1non_trainable_variables
	variables
2layer_regularization_losses
trainable_variables
3metrics
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
4layer_metrics

5layers
regularization_losses
6non_trainable_variables
	variables
7layer_regularization_losses
trainable_variables
8metrics
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
?	

$kernel
%recurrent_kernel
&bias
'input_gate_peephole_weights
 (forget_gate_peephole_weights
 )output_gate_peephole_weights
9regularization_losses
:	variables
;trainable_variables
<	keras_api
g__call__
*h&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Addons>PeepholeLSTMCell", "name": "peephole_lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "peephole_lstm_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
$0
%1
&2
'3
(4
)5"
trackable_list_wrapper
J
$0
%1
&2
'3
(4
)5"
trackable_list_wrapper
?
=layer_metrics

>states

?layers
regularization_losses
@non_trainable_variables
	variables
Alayer_regularization_losses
trainable_variables
Bmetrics
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Clayer_metrics

Dlayers
regularization_losses
Enon_trainable_variables
	variables
Flayer_regularization_losses
trainable_variables
Gmetrics
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
0:.	 ?2rnn/peephole_lstm_cell/kernel
::8	 ?2'rnn/peephole_lstm_cell/recurrent_kernel
*:(?2rnn/peephole_lstm_cell/bias
@:> 22rnn/peephole_lstm_cell/input_gate_peephole_weights
A:? 23rnn/peephole_lstm_cell/forget_gate_peephole_weights
A:? 23rnn/peephole_lstm_cell/output_gate_peephole_weights
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
$0
%1
&2
'3
(4
)5"
trackable_list_wrapper
J
$0
%1
&2
'3
(4
)5"
trackable_list_wrapper
?
Ilayer_metrics

Jlayers
9regularization_losses
Knon_trainable_variables
:	variables
Llayer_regularization_losses
;trainable_variables
Mmetrics
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Ntotal
	Ocount
P	variables
Q	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
N0
O1"
trackable_list_wrapper
-
P	variables"
_generic_user_object
4:2	? 2#SGD/embedding_2/embeddings/momentum
+:) 2SGD/dense_3/kernel/momentum
%:#2SGD/dense_3/bias/momentum
;:9	 ?2*SGD/rnn/peephole_lstm_cell/kernel/momentum
E:C	 ?24SGD/rnn/peephole_lstm_cell/recurrent_kernel/momentum
5:3?2(SGD/rnn/peephole_lstm_cell/bias/momentum
K:I 2?SGD/rnn/peephole_lstm_cell/input_gate_peephole_weights/momentum
L:J 2@SGD/rnn/peephole_lstm_cell/forget_gate_peephole_weights/momentum
L:J 2@SGD/rnn/peephole_lstm_cell/output_gate_peephole_weights/momentum
?2?
,__inference_sequential_2_layer_call_fn_29642
,__inference_sequential_2_layer_call_fn_29179
,__inference_sequential_2_layer_call_fn_29665
,__inference_sequential_2_layer_call_fn_29130?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_27759?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
embedding_2_input?????????2
?2?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29619
G__inference_sequential_2_layer_call_and_return_conditional_losses_29080
G__inference_sequential_2_layer_call_and_return_conditional_losses_29423
G__inference_sequential_2_layer_call_and_return_conditional_losses_29054?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_embedding_2_layer_call_fn_29682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_embedding_2_layer_call_and_return_conditional_losses_29675?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_spatial_dropout1d_2_layer_call_fn_29756
3__inference_spatial_dropout1d_2_layer_call_fn_29719
3__inference_spatial_dropout1d_2_layer_call_fn_29714
3__inference_spatial_dropout1d_2_layer_call_fn_29751?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29704
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29709
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29746
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29741?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_rnn_layer_call_fn_30137
#__inference_rnn_layer_call_fn_30154
#__inference_rnn_layer_call_fn_30552
#__inference_rnn_layer_call_fn_30535?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_rnn_layer_call_and_return_conditional_losses_30336
>__inference_rnn_layer_call_and_return_conditional_losses_30518
>__inference_rnn_layer_call_and_return_conditional_losses_29938
>__inference_rnn_layer_call_and_return_conditional_losses_30120?
???
FullArgSpecO
argsG?D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults?

 
p 

 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_3_layer_call_fn_30572?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_30563?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_29210embedding_2_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_peephole_lstm_cell_layer_call_fn_30685
2__inference_peephole_lstm_cell_layer_call_fn_30708?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_30617
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_30662?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_27759z	$%&'():?7
0?-
+?(
embedding_2_input?????????2
? "1?.
,
dense_3!?
dense_3??????????
B__inference_dense_3_layer_call_and_return_conditional_losses_30563\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_dense_3_layer_call_fn_30572O/?,
%?"
 ?
inputs????????? 
? "???????????
F__inference_embedding_2_layer_call_and_return_conditional_losses_29675_/?,
%?"
 ?
inputs?????????2
? ")?&
?
0?????????2 
? ?
+__inference_embedding_2_layer_call_fn_29682R/?,
%?"
 ?
inputs?????????2
? "??????????2 ?
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_30617?$%&'()??}
v?s
 ?
inputs????????? 
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
M__inference_peephole_lstm_cell_layer_call_and_return_conditional_losses_30662?$%&'()??}
v?s
 ?
inputs????????? 
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
2__inference_peephole_lstm_cell_layer_call_fn_30685?$%&'()??}
v?s
 ?
inputs????????? 
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
2__inference_peephole_lstm_cell_layer_call_fn_30708?$%&'()??}
v?s
 ?
inputs????????? 
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
>__inference_rnn_layer_call_and_return_conditional_losses_29938t$%&'()C?@
9?6
$?!
inputs?????????2 

 
p

 

 
? "%?"
?
0????????? 
? ?
>__inference_rnn_layer_call_and_return_conditional_losses_30120t$%&'()C?@
9?6
$?!
inputs?????????2 

 
p 

 

 
? "%?"
?
0????????? 
? ?
>__inference_rnn_layer_call_and_return_conditional_losses_30336?$%&'()S?P
I?F
4?1
/?,
inputs/0?????????????????? 

 
p

 

 
? "%?"
?
0????????? 
? ?
>__inference_rnn_layer_call_and_return_conditional_losses_30518?$%&'()S?P
I?F
4?1
/?,
inputs/0?????????????????? 

 
p 

 

 
? "%?"
?
0????????? 
? ?
#__inference_rnn_layer_call_fn_30137g$%&'()C?@
9?6
$?!
inputs?????????2 

 
p

 

 
? "?????????? ?
#__inference_rnn_layer_call_fn_30154g$%&'()C?@
9?6
$?!
inputs?????????2 

 
p 

 

 
? "?????????? ?
#__inference_rnn_layer_call_fn_30535w$%&'()S?P
I?F
4?1
/?,
inputs/0?????????????????? 

 
p

 

 
? "?????????? ?
#__inference_rnn_layer_call_fn_30552w$%&'()S?P
I?F
4?1
/?,
inputs/0?????????????????? 

 
p 

 

 
? "?????????? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29054v	$%&'()B??
8?5
+?(
embedding_2_input?????????2
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29080v	$%&'()B??
8?5
+?(
embedding_2_input?????????2
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29423k	$%&'()7?4
-?*
 ?
inputs?????????2
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29619k	$%&'()7?4
-?*
 ?
inputs?????????2
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_2_layer_call_fn_29130i	$%&'()B??
8?5
+?(
embedding_2_input?????????2
p

 
? "???????????
,__inference_sequential_2_layer_call_fn_29179i	$%&'()B??
8?5
+?(
embedding_2_input?????????2
p 

 
? "???????????
,__inference_sequential_2_layer_call_fn_29642^	$%&'()7?4
-?*
 ?
inputs?????????2
p

 
? "???????????
,__inference_sequential_2_layer_call_fn_29665^	$%&'()7?4
-?*
 ?
inputs?????????2
p 

 
? "???????????
#__inference_signature_wrapper_29210?	$%&'()O?L
? 
E?B
@
embedding_2_input+?(
embedding_2_input?????????2"1?.
,
dense_3!?
dense_3??????????
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29704?I?F
??<
6?3
inputs'???????????????????????????
p
? ";?8
1?.
0'???????????????????????????
? ?
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29709?I?F
??<
6?3
inputs'???????????????????????????
p 
? ";?8
1?.
0'???????????????????????????
? ?
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29741d7?4
-?*
$?!
inputs?????????2 
p
? ")?&
?
0?????????2 
? ?
N__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_29746d7?4
-?*
$?!
inputs?????????2 
p 
? ")?&
?
0?????????2 
? ?
3__inference_spatial_dropout1d_2_layer_call_fn_29714{I?F
??<
6?3
inputs'???????????????????????????
p
? ".?+'????????????????????????????
3__inference_spatial_dropout1d_2_layer_call_fn_29719{I?F
??<
6?3
inputs'???????????????????????????
p 
? ".?+'????????????????????????????
3__inference_spatial_dropout1d_2_layer_call_fn_29751W7?4
-?*
$?!
inputs?????????2 
p
? "??????????2 ?
3__inference_spatial_dropout1d_2_layer_call_fn_29756W7?4
-?*
$?!
inputs?????????2 
p 
? "??????????2 