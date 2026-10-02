/* Pre-initialization ABI check: does not discover or activate GPU devices. */
#include <hsa/hsa.h>
#include <stdio.h>

int main(void) {
  const char *description = NULL;
  hsa_status_t status = hsa_status_string(HSA_STATUS_SUCCESS, &description);
  if (status != HSA_STATUS_SUCCESS || description == NULL || description[0] == '\0') {
    fprintf(stderr, "HSA status-string ABI check failed\n");
    return 1;
  }
  return 0;
}
