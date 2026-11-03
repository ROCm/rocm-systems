#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x);if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));exit(1);} } while(0)
int main() {
 unsigned char *a,*b; hipStream_t stream;
 CHECK(hipMalloc(&a,256));CHECK(hipMalloc(&b,256));CHECK(hipStreamCreate(&stream));
 std::vector<unsigned char> src(256),got(256);for(int i=0;i<256;++i)src[i]=(i*7+3)%251;
 CHECK(hipMemcpy(a,src.data(),256,hipMemcpyHostToDevice));
 auto run=[&](hipGraphExec_t exec,int dest,bool duplicate){
  for(int t=0;t<3;++t){
   CHECK(hipMemsetAsync(b,0,256,stream));CHECK(hipGraphLaunch(exec,stream));CHECK(hipStreamSynchronize(stream));
   CHECK(hipMemcpy(got.data(),b,256,hipMemcpyDeviceToHost));
   std::vector<unsigned char> expected(256,0);
   for(int i=0;i<33;++i)expected[dest+i]=src[1+i];
   if(!duplicate)for(int i=0;i<32;++i)expected[64+i]=src[64+i];
   if(got!=expected){fprintf(stderr,"GPU copy oracle failed\n");exit(2);}
  }
 };
 hipGraph_t graph;hipGraphExec_t exec,fresh;
 CHECK(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));
 CHECK(hipMemcpyAsync(b+3,a+1,33,hipMemcpyDeviceToDevice,stream));
 CHECK(hipMemcpyAsync(b+64,a+64,32,hipMemcpyDeviceToDevice,stream));
 CHECK(hipStreamEndCapture(stream,&graph));CHECK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));run(exec,3,false);
 size_t n=0;CHECK(hipGraphGetNodes(graph,nullptr,&n));std::vector<hipGraphNode_t> nodes(n);CHECK(hipGraphGetNodes(graph,nodes.data(),&n));
 hipGraphNode_t target=nullptr;
 for(auto node:nodes){hipMemcpy3DParms p{};CHECK(hipGraphMemcpyNodeGetParams(node,&p));if(p.srcPtr.ptr==a+1)target=node;}
 if(!target)return 3;
 CHECK(hipGraphExecMemcpyNodeSetParams1D(exec,target,b+128,a+1,33,hipMemcpyDeviceToDevice));run(exec,128,false);
 CHECK(hipGraphInstantiate(&fresh,graph,nullptr,nullptr,0));run(fresh,3,false);
 CHECK(hipGraphExecDestroy(exec));CHECK(hipGraphExecDestroy(fresh));CHECK(hipGraphDestroy(graph));
 CHECK(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));
 CHECK(hipMemcpyAsync(b+3,a+1,33,hipMemcpyDeviceToDevice,stream));
 CHECK(hipMemcpyAsync(b+3,a+1,33,hipMemcpyDeviceToDevice,stream));
 CHECK(hipStreamEndCapture(stream,&graph));CHECK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));run(exec,3,true);
 CHECK(hipGraphExecDestroy(exec));CHECK(hipGraphDestroy(graph));CHECK(hipStreamDestroy(stream));CHECK(hipFree(a));CHECK(hipFree(b));
 puts("PASS: copy GPU contents including offsets, unaligned size, mutation and duplicates");
}
