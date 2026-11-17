; Reduced from PyTorch's warpMergeSortTopK<double> kernel.
; This is a code-generation reproducer and is not intended to be executed.

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0

define amdgpu_kernel void @repro_spill(i32 %0, i32 %1, i32 %2, double %3, i1 %4, double %5, i1 %6, i1 %7, double %8, i1 %9, double %10, i1 %11, i1 %12, double %13, i1 %14, double %15, i1 %16, double %17, double %18, i1 %19, i1 %20, <16 x double> %21, i1 %22, i1 %23, i1 %24, i1 %25, double %26, i1 %27, i1 %28, double %29, i1 %30, i1 %31, i1 %32, i1 %33, i64 %34, double %35, i1 %36, i1 %37, i1 %38, double %39, ptr addrspace(3) %40, ptr addrspace(3) %41, ptr addrspace(3) %42, double %43, i1 %44, i1 %45, ptr addrspace(3) %46, i32 %47) {
  %49 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %50 = udiv i32 %49, %1
  %51 = freeze i32 %2
  %52 = udiv i32 %50, %51
  %53 = zext i32 %52 to i64
  %54 = getelementptr [8 x i8], ptr addrspace(1) null, i64 %53
  %55 = mul i32 %49, %47
  %56 = udiv i32 %55, %51
  %57 = zext i32 %56 to i64
  %58 = getelementptr [8 x i8], ptr addrspace(1) null, i64 %57
  %59 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %60 = zext i32 %59 to i64
  %61 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 0, i32 0)
  %62 = insertelement <16 x double> zeroinitializer, double %3, i64 4
  %63 = load double, ptr addrspace(3) null, align 8
  %64 = insertelement <16 x double> %62, double %63, i64 5
  %65 = insertelement <16 x double> %64, double 0.000000e+00, i64 0
  %66 = insertelement <16 x double> %65, double 0.000000e+00, i64 0
  %67 = load double, ptr addrspace(3) inttoptr (i32 64 to ptr addrspace(3)), align 8
  %68 = insertelement <16 x double> %66, double %67, i64 8
  %69 = insertelement <16 x double> %68, double 0.000000e+00, i64 0
  %70 = insertelement <16 x double> %69, double 0.000000e+00, i64 0
  %71 = insertelement <16 x double> %70, double 0.000000e+00, i64 0
  %72 = insertelement <16 x double> %71, double 0.000000e+00, i64 0
  %73 = insertelement <16 x double> %72, double 0.000000e+00, i64 0
  %74 = insertelement <16 x double> %73, double 0.000000e+00, i64 0
  %75 = insertelement <16 x double> %74, double 0.000000e+00, i64 0
  %76 = shl i32 %61, 1
  %77 = icmp slt i32 %76, %0
  %78 = select i1 %77, double 0.000000e+00, double +qnan
  %79 = insertelement <16 x double> %75, double +qnan, i64 1
  %80 = select i1 %77, <16 x double> %75, <16 x double> %79
  %81 = extractelement <16 x double> %80, i64 4
  %82 = fcmp olt double 0.000000e+00, %81
  %83 = select i1 %82, double %81, double 0.000000e+00
  br i1 %4, label %86, label %84

84:                                               ; preds = %48
  %85 = insertelement <16 x double> %80, double %83, i64 1
  br label %92

86:                                               ; preds = %48
  %87 = fcmp ord double %83, 0.000000e+00
  %88 = select i1 %6, i1 %87, i1 false
  %89 = fcmp olt double %83, %5
  %90 = select i1 %88, i1 false, i1 %89
  %91 = select i1 %90, double 0.000000e+00, double %83
  br label %92

92:                                               ; preds = %86, %84
  %93 = phi <16 x double> [ %80, %86 ], [ %85, %84 ]
  %94 = phi double [ %91, %86 ], [ %83, %84 ]
  br i1 %7, label %common.ret, label %95

95:                                               ; preds = %92
  %96 = insertelement <16 x double> %93, double %94, i64 6
  %97 = fcmp uno double %8, 0.000000e+00
  %98 = fcmp ord double %94, 0.000000e+00
  %99 = select i1 %97, i1 %98, i1 false
  %100 = select i1 %99, i1 false, i1 %9
  %101 = select i1 %100, double 0.000000e+00, double %94
  %102 = icmp slt i32 %76, 1
  br i1 %102, label %105, label %103

common.ret:                                       ; preds = %370, %92
  ret void

103:                                              ; preds = %95
  %104 = insertelement <16 x double> %96, double 0.000000e+00, i64 8
  br label %108

105:                                              ; preds = %95
  %106 = extractelement <16 x double> %96, i64 8
  %107 = fcmp uno double 0.000000e+00, 0.000000e+00
  br label %108

108:                                              ; preds = %105, %103
  %109 = phi <16 x double> [ %96, %105 ], [ %104, %103 ]
  %110 = phi double [ %106, %105 ], [ %101, %103 ]
  %111 = icmp slt i32 %76, 1
  br i1 %111, label %114, label %112

112:                                              ; preds = %108
  %113 = insertelement <16 x double> %109, double 0.000000e+00, i64 0
  br label %121

114:                                              ; preds = %108
  %115 = extractelement <16 x double> %109, i64 10
  %116 = fcmp olt double 0.000000e+00, 0.000000e+00
  %117 = fcmp uno double %10, 0.000000e+00
  %118 = select i1 %117, i1 %11, i1 false
  %119 = select i1 %118, i1 false, i1 %12
  %120 = select i1 %119, double 1.000000e+00, double 0.000000e+00
  br label %121

121:                                              ; preds = %114, %112
  %122 = phi <16 x double> [ %109, %114 ], [ %113, %112 ]
  %123 = phi double [ %115, %114 ], [ 0.000000e+00, %112 ]
  %124 = phi double [ %120, %114 ], [ 0.000000e+00, %112 ]
  %125 = icmp slt i32 %76, 1
  br i1 %125, label %128, label %126

126:                                              ; preds = %121
  %127 = insertelement <16 x double> %122, double 0.000000e+00, i64 0
  br label %135

128:                                              ; preds = %121
  %129 = extractelement <16 x double> %122, i64 0
  %130 = fcmp uno double %13, 0.000000e+00
  %131 = select i1 %130, i1 true, i1 false
  %132 = fcmp olt double %124, %13
  %133 = select i1 %131, i1 false, i1 %132
  %134 = select i1 %133, double %13, double 0.000000e+00
  br label %135

135:                                              ; preds = %128, %126
  %136 = phi <16 x double> [ %122, %128 ], [ %127, %126 ]
  %137 = phi double [ %129, %128 ], [ 0.000000e+00, %126 ]
  %138 = phi double [ %134, %128 ], [ 0.000000e+00, %126 ]
  %139 = icmp slt i32 %76, 1
  br i1 %139, label %142, label %140

140:                                              ; preds = %135
  %141 = insertelement <16 x double> %136, double 0.000000e+00, i64 0
  br label %149

142:                                              ; preds = %135
  %143 = extractelement <16 x double> %136, i64 0
  %144 = fcmp uno double %43, 0.000000e+00
  %145 = select i1 %144, i1 %14, i1 false
  %146 = fcmp olt double %138, 1.000000e+00
  %147 = select i1 %145, i1 false, i1 %146
  %148 = select i1 %147, double 1.000000e+00, double 0.000000e+00
  br label %149

149:                                              ; preds = %142, %140
  %150 = phi <16 x double> [ %136, %142 ], [ %141, %140 ]
  %151 = phi double [ %143, %142 ], [ 0.000000e+00, %140 ]
  %152 = phi double [ %148, %142 ], [ 0.000000e+00, %140 ]
  %153 = insertelement <16 x double> %150, double %152, i64 15
  %154 = select i1 %4, <16 x double> %150, <16 x double> %153
  %155 = icmp slt i32 %76, 1
  br i1 %155, label %156, label %370

156:                                              ; preds = %149
  %157 = extractelement <16 x double> %154, i64 0
  %158 = fcmp olt double 0.000000e+00, %157
  br i1 %158, label %339, label %342

159:                                              ; preds = %342
  %160 = shufflevector <16 x double> %368, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 2, i32 poison, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %161 = insertelement <16 x double> %160, double 0.000000e+00, i64 2
  br label %162

162:                                              ; preds = %342, %159
  %163 = phi i64 [ 1, %159 ], [ 0, %342 ]
  %164 = phi i64 [ 0, %159 ], [ 1, %342 ]
  %165 = phi <16 x double> [ %161, %159 ], [ %368, %342 ]
  %166 = phi double [ %346, %159 ], [ 0.000000e+00, %342 ]
  %167 = fcmp uno double %15, 0.000000e+00
  %168 = shufflevector <16 x double> %165, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8, i32 7, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %169 = select i1 %167, <16 x double> %168, <16 x double> %165
  %170 = shufflevector <16 x double> %169, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 14, i32 13, i32 15>
  %171 = select i1 %16, <16 x double> %170, <16 x double> %169
  %172 = extractelement <16 x double> %171, i64 1
  %173 = fcmp olt double %172, 0.000000e+00
  br i1 %173, label %174, label %177

174:                                              ; preds = %162
  %175 = shufflevector <16 x double> %171, <16 x double> zeroinitializer, <16 x i32> <i32 1, i32 poison, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %176 = insertelement <16 x double> %175, double %345, i64 1
  br label %177

177:                                              ; preds = %174, %162
  %178 = phi i64 [ %164, %174 ], [ %343, %162 ]
  %179 = phi <16 x double> [ %176, %174 ], [ %171, %162 ]
  %180 = phi double [ 0.000000e+00, %174 ], [ %17, %162 ]
  %181 = phi double [ 0.000000e+00, %174 ], [ %18, %162 ]
  %182 = shufflevector <16 x double> %179, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 3, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %183 = insertelement <16 x double> %182, double %166, i64 3
  %184 = select i1 %19, <16 x double> %183, <16 x double> %179
  %185 = shufflevector <16 x double> %184, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %186 = select i1 %20, <16 x double> %185, <16 x double> %184
  %187 = extractelement <16 x double> %186, i64 13
  %188 = fcmp olt double %187, 0.000000e+00
  %189 = shufflevector <16 x double> %186, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 13, i32 12, i32 14, i32 15>
  %190 = select i1 %188, <16 x double> %189, <16 x double> %186
  %191 = fcmp olt double 0.000000e+00, %181
  br i1 %191, label %192, label %193

192:                                              ; preds = %177
  br label %193

193:                                              ; preds = %192, %177
  %194 = phi i64 [ 0, %192 ], [ 1, %177 ]
  %195 = phi i64 [ 1, %192 ], [ 0, %177 ]
  %196 = phi <16 x double> [ zeroinitializer, %192 ], [ %190, %177 ]
  %197 = phi double [ 1.000000e+00, %192 ], [ 0.000000e+00, %177 ]
  %198 = shufflevector <16 x double> %196, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 4, i32 3, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %199 = select i1 %22, <16 x double> %198, <16 x double> %196
  %200 = extractelement <16 x double> %199, i64 13
  %201 = shufflevector <16 x double> %199, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 14, i32 13, i32 15>
  %202 = shufflevector <16 x double> %201, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 3, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %203 = fcmp uno double %200, 0.000000e+00
  %204 = zext i32 %59 to i64
  %205 = select i1 %203, i64 %204, i64 0
  %206 = insertelement <16 x double> %202, double %197, i64 3
  %207 = select i1 %23, <16 x double> %206, <16 x double> %201
  %208 = shufflevector <16 x double> %207, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 13, i32 12, i32 14, i32 15>
  %209 = select i1 %24, <16 x double> %208, <16 x double> %207
  %210 = extractelement <16 x double> %209, i64 6
  %211 = fcmp olt double %26, 0.000000e+00
  %212 = shufflevector <16 x double> %209, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 10, i32 9, i32 11, i32 12, i32 13, i32 14, i32 15>
  %213 = select i1 %211, <16 x double> %212, <16 x double> %209
  %214 = extractelement <16 x double> %213, i64 1
  %215 = fcmp olt double %214, 0.000000e+00
  br i1 %215, label %216, label %219

216:                                              ; preds = %193
  %217 = shufflevector <16 x double> %213, <16 x double> zeroinitializer, <16 x i32> <i32 1, i32 poison, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %218 = insertelement <16 x double> %217, double %180, i64 1
  br label %219

219:                                              ; preds = %216, %193
  %220 = phi i64 [ %178, %216 ], [ 0, %193 ]
  %221 = phi i64 [ %195, %216 ], [ %178, %193 ]
  %222 = phi <16 x double> [ %218, %216 ], [ %213, %193 ]
  %223 = phi double [ %214, %216 ], [ 0.000000e+00, %193 ]
  %224 = phi double [ %180, %216 ], [ %214, %193 ]
  %225 = select i1 %22, i64 %163, i64 0
  %226 = select i1 true, i64 %225, i64 0
  %227 = select i1 %27, i64 %226, i64 %194
  %228 = shufflevector <16 x double> %222, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 3, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %229 = insertelement <16 x double> %228, double 0.000000e+00, i64 3
  %230 = select i1 %27, <16 x double> %229, <16 x double> %222
  %231 = extractelement <16 x double> %230, i64 7
  %232 = fcmp olt double %210, 0.000000e+00
  %233 = shufflevector <16 x double> %230, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 15, i32 14>
  %234 = select i1 %28, <16 x double> %233, <16 x double> %230
  %235 = fcmp uno double %224, 0.000000e+00
  br i1 %235, label %236, label %239

236:                                              ; preds = %219
  %237 = shufflevector <16 x double> %234, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 2, i32 poison, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %238 = insertelement <16 x double> %237, double 0.000000e+00, i64 2
  br label %239

239:                                              ; preds = %236, %219
  %240 = phi i64 [ 0, %236 ], [ %227, %219 ]
  %241 = phi i64 [ 0, %236 ], [ %220, %219 ]
  %242 = phi <16 x double> [ %238, %236 ], [ %234, %219 ]
  %243 = phi double [ 0.000000e+00, %236 ], [ %29, %219 ]
  %244 = shufflevector <16 x double> %242, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 4, i32 3, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %245 = select i1 %30, <16 x double> %244, <16 x double> %242
  %246 = shufflevector <16 x double> %245, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 8, i32 7, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %247 = select i1 %31, <16 x double> %246, <16 x double> %245
  %248 = extractelement <16 x double> %247, i64 0
  %249 = fcmp olt double %248, 0.000000e+00
  %250 = shufflevector <16 x double> %247, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 12, i32 11, i32 13, i32 14, i32 15>
  %251 = select i1 %249, <16 x double> %250, <16 x double> %247
  %252 = shufflevector <16 x double> %251, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 14, i32 13, i32 15>
  %253 = select i1 %32, <16 x double> %252, <16 x double> %251
  %254 = extractelement <16 x double> %253, i64 1
  %255 = fcmp olt double %254, 0.000000e+00
  br i1 %255, label %256, label %259

256:                                              ; preds = %239
  %257 = shufflevector <16 x double> %253, <16 x double> zeroinitializer, <16 x i32> <i32 1, i32 poison, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %258 = insertelement <16 x double> %257, double %223, i64 1
  br label %259

259:                                              ; preds = %256, %239
  %260 = phi i64 [ %241, %256 ], [ %221, %239 ]
  %261 = phi <16 x double> [ %258, %256 ], [ %253, %239 ]
  %262 = phi double [ 0.000000e+00, %256 ], [ 1.000000e+00, %239 ]
  %263 = phi double [ %223, %256 ], [ %254, %239 ]
  %264 = select i1 %33, i64 0, i64 %240
  %265 = fcmp olt double 0.000000e+00, %243
  %266 = shufflevector <16 x double> %261, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 3, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %267 = insertelement <16 x double> %266, double 0.000000e+00, i64 3
  %268 = fcmp olt double %231, 0.000000e+00
  %269 = select i1 %232, i64 %225, i64 0
  %270 = select i1 %268, i64 %269, i64 0
  %271 = select i1 %265, <16 x double> %267, <16 x double> %261
  %272 = shufflevector <16 x double> %271, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 9, i32 8, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %273 = select i1 %44, <16 x double> %272, <16 x double> %271
  br i1 %249, label %274, label %277

274:                                              ; preds = %259
  %275 = shufflevector <16 x double> %273, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 13, i32 poison, i32 14, i32 15>
  %276 = insertelement <16 x double> %275, double 0.000000e+00, i64 0
  br label %277

277:                                              ; preds = %274, %259
  %278 = phi i64 [ 0, %274 ], [ %205, %259 ]
  %279 = phi i64 [ 1, %274 ], [ 0, %259 ]
  %280 = phi <16 x double> [ %276, %274 ], [ %273, %259 ]
  %281 = phi double [ 1.000000e+00, %274 ], [ 0.000000e+00, %259 ]
  %282 = fcmp olt double %35, 0.000000e+00
  %283 = shufflevector <16 x double> %280, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 15, i32 14>
  %284 = select i1 %282, <16 x double> %283, <16 x double> %280
  %285 = extractelement <16 x double> %284, i64 2
  %286 = fcmp uno double %263, 0.000000e+00
  br i1 %286, label %287, label %289

287:                                              ; preds = %277
  %288 = insertelement <16 x double> zeroinitializer, double %263, i64 2
  br label %289

289:                                              ; preds = %287, %277
  %290 = phi i64 [ 0, %287 ], [ %264, %277 ]
  %291 = phi i64 [ 1, %287 ], [ 0, %277 ]
  %292 = phi <16 x double> [ %288, %287 ], [ %284, %277 ]
  %293 = phi double [ %263, %287 ], [ %285, %277 ]
  %294 = extractelement <16 x double> %292, i64 0
  %295 = fcmp olt double %294, 0.000000e+00
  %296 = shufflevector <16 x double> %292, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 10, i32 9, i32 11, i32 12, i32 13, i32 14, i32 15>
  %297 = select i1 %295, <16 x double> %296, <16 x double> %292
  %298 = extractelement <16 x double> %297, i64 0
  %299 = fcmp olt double %298, 0.000000e+00
  %300 = shufflevector <16 x double> %297, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 12, i32 11, i32 13, i32 14, i32 15>
  %301 = select i1 %299, <16 x double> %300, <16 x double> %297
  %302 = fcmp olt double 0.000000e+00, %281
  br i1 %302, label %303, label %304

303:                                              ; preds = %289
  br label %304

304:                                              ; preds = %303, %289
  %305 = phi i64 [ 1, %303 ], [ 0, %289 ]
  %306 = phi i64 [ 0, %303 ], [ %278, %289 ]
  %307 = phi <16 x double> [ zeroinitializer, %303 ], [ %301, %289 ]
  %308 = phi double [ %281, %303 ], [ 0.000000e+00, %289 ]
  %309 = extractelement <16 x double> %307, i64 1
  %310 = fcmp olt double %309, 0.000000e+00
  br i1 %310, label %311, label %314

311:                                              ; preds = %304
  %312 = shufflevector <16 x double> %307, <16 x double> zeroinitializer, <16 x i32> <i32 1, i32 poison, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %313 = insertelement <16 x double> %312, double %262, i64 1
  br label %314

314:                                              ; preds = %311, %304
  %315 = phi i64 [ 1, %311 ], [ 0, %304 ]
  %316 = phi i64 [ %291, %311 ], [ %260, %304 ]
  %317 = phi <16 x double> [ %313, %311 ], [ %307, %304 ]
  %318 = phi double [ 0.000000e+00, %311 ], [ %309, %304 ]
  %319 = fcmp olt double 0.000000e+00, %293
  %320 = select i1 %319, i64 0, i64 %290
  %321 = select i1 %36, <16 x double> zeroinitializer, <16 x double> %317
  %322 = extractelement <16 x double> %321, i64 11
  %323 = fcmp olt double %322, 0.000000e+00
  %324 = select i1 %323, i64 %270, i64 %279
  %325 = extractelement <16 x double> %321, i64 13
  %326 = fcmp olt double %325, 0.000000e+00
  %327 = select i1 %326, i64 %306, i64 0
  %328 = select i1 %326, <16 x double> zeroinitializer, <16 x double> %321
  %329 = extractelement <16 x double> %328, i64 15
  %330 = fcmp olt double %329, %308
  %331 = select i1 %330, i64 0, i64 %305
  %332 = fcmp olt double 0.000000e+00, %318
  %333 = select i1 %332, i64 %320, i64 %315
  %334 = extractelement <16 x double> %328, i64 0
  %335 = fcmp uno double %334, 0.000000e+00
  %336 = shufflevector <16 x double> %328, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 4, i32 3, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %337 = select i1 %335, <16 x double> %336, <16 x double> %328
  %338 = shufflevector <16 x double> %337, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 14, i32 13, i32 15>
  br label %370

339:                                              ; preds = %156
  %340 = insertelement <16 x double> %154, double 0.000000e+00, i64 0
  %341 = shufflevector <16 x double> %340, <16 x double> %154, <16 x i32> <i32 0, i32 16, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  br label %342

342:                                              ; preds = %339, %156
  %343 = phi i64 [ 0, %339 ], [ 1, %156 ]
  %344 = phi <16 x double> [ %341, %339 ], [ %154, %156 ]
  %345 = phi double [ %78, %339 ], [ %157, %156 ]
  %346 = phi double [ 1.000000e+00, %339 ], [ 0.000000e+00, %156 ]
  %347 = insertelement <16 x double> %344, double 0.000000e+00, i64 2
  %348 = insertelement <16 x double> %347, double 0.000000e+00, i64 0
  %349 = shufflevector <16 x double> %348, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 poison, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %350 = insertelement <16 x double> %349, double %81, i64 0
  %351 = select i1 %82, <16 x double> %350, <16 x double> %348
  %352 = fcmp olt double 0.000000e+00, %94
  %353 = shufflevector <16 x double> %351, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 7, i32 poison, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %354 = insertelement <16 x double> %353, double 0.000000e+00, i64 0
  %355 = select i1 %352, <16 x double> %354, <16 x double> %351
  %356 = fcmp olt double 0.000000e+00, %110
  %357 = shufflevector <16 x double> %355, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 9, i32 poison, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %358 = insertelement <16 x double> %357, double %110, i64 0
  %359 = select i1 %356, <16 x double> %358, <16 x double> %355
  %360 = shufflevector <16 x double> %359, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 11, i32 poison, i32 12, i32 13, i32 14, i32 15>
  %361 = insertelement <16 x double> %360, double %123, i64 11
  %362 = select i1 %37, <16 x double> %361, <16 x double> %359
  %363 = shufflevector <16 x double> %362, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 13, i32 poison, i32 14, i32 15>
  %364 = insertelement <16 x double> %363, double %137, i64 13
  %365 = select i1 %38, <16 x double> %364, <16 x double> %362
  %366 = shufflevector <16 x double> %365, <16 x double> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 15, i32 poison>
  %367 = insertelement <16 x double> %366, double %151, i64 15
  %368 = select i1 %45, <16 x double> %367, <16 x double> %365
  %369 = fcmp olt double 0.000000e+00, %346
  br i1 %369, label %159, label %162

370:                                              ; preds = %314, %149
  %371 = phi i64 [ %331, %314 ], [ 0, %149 ]
  %372 = phi i64 [ %327, %314 ], [ 0, %149 ]
  %373 = phi i64 [ %324, %314 ], [ 0, %149 ]
  %374 = phi i64 [ %333, %314 ], [ 0, %149 ]
  %375 = phi i64 [ %316, %314 ], [ %60, %149 ]
  %376 = phi <16 x double> [ %338, %314 ], [ %154, %149 ]
  %377 = extractelement <16 x double> %376, i64 1
  store double %377, ptr addrspace(3) %46, align 8
  %378 = extractelement <16 x double> %376, i64 2
  %379 = getelementptr i8, ptr addrspace(3) %40, i32 16
  store double %378, ptr addrspace(3) %379, align 8
  %380 = extractelement <16 x double> %376, i64 3
  %381 = getelementptr i8, ptr addrspace(3) %40, i32 24
  store double %380, ptr addrspace(3) %381, align 8
  %382 = extractelement <16 x double> %376, i64 10
  store double %382, ptr addrspace(3) null, align 8
  fence acquire
  %383 = getelementptr [8 x i8], ptr addrspace(3) null, i32 %76
  store i64 %375, ptr addrspace(3) %383, align 8
  %384 = getelementptr i8, ptr addrspace(3) %383, i32 8
  store i64 %374, ptr addrspace(3) %384, align 8
  store i64 %373, ptr addrspace(3) %41, align 8
  store i64 %372, ptr addrspace(3) %42, align 8
  store i64 %371, ptr addrspace(3) null, align 8
  store double 0.000000e+00, ptr addrspace(1) %54, align 8
  store i64 0, ptr addrspace(1) %58, align 8
  br label %common.ret

; uselistorder directives
  uselistorder <16 x double> %368, { 1, 0 }
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y() #1

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
