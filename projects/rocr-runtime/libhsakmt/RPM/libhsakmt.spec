%define name        hsakmt-rocm-dev
%define version     %{getenv:PACKAGE_VER}
%define packageroot %{getenv:PACKAGE_DIR}

Name:       %{name}
Version:    %{version}
Release:    1
Summary:    Thunk libraries for AMD KFD

Group:      System Environment/Libraries
License:    Advanced Micro Devices Inc.

%if 0%{?centos} == 6
Requires:   numactl
%else
Requires:   numactl-libs
%endif


%description
This package includes the libhsakmt (Thunk) libraries
for AMD KFD

%prep
%setup -T -D -c -n %{name}

%install
cp -R %packageroot $RPM_BUILD_ROOT
find $RPM_BUILD_ROOT \! -type d | sed "s|$RPM_BUILD_ROOT||"> thunk.list

%post
ldconfig

%postun
ldconfig

%clean
rm -rf $RPM_BUILD_ROOT

%files -f thunk.list

%defattr(-,root,root,-)
