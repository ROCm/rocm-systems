.. meta::
  :description: Test page for verifying the version banner latest-page fallback behavior.
  :keywords: AMD, ROCm, HIP, banner, test

*******************************************************************************
Banner fallback test
*******************************************************************************

This page exists only to test the version banner's latest-page fallback. It is
new on this branch and does not exist on the latest version of the HIP
documentation.

When this branch is served as an old version, the banner at the top of this
page links to the same page under ``latest``. Because this page does not exist
there and no redirect points to it, clicking the banner should send you to the
HIP documentation landing page on ``latest`` instead of a 404.
