# FAQ

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

**1. How do I export profiling data I have already generated using Omniperf?**

In order to interact with the Grafana GUI you must sync data with the MongoDB backend. This interaction is done through ***database*** mode.

Simply pass the directory of your desired workload like so,
```shell
$ omniperf database --import -w <path-to-results> -H <hostname> -u <username> -t <team-name>
```
**2. python ast error: 'Constant' object has no attribute 'kind'**

This comes from a bug in the default astunparse 1.6.3 with python 3.8. Seems good with python 3.7 and 3.9.

Workaround:
```shell
$ pip3 uninstall astunparse
$ pip3 astunparse
```

**3. tabulate doesn't print properly**
Workaround:
```shell
$ export LC_ALL=C.UTF-8
$ export LANG=C.UTF-8
```

**3. How can I SSH Tunnel in MobaXterm?**

1. Open MobaXterm
2. In the top ribbon, select `Tunneling`
``` {image} images/tunnel_demo1.png
:alt: MobaXterm Tunnel Button
:class: bg-primary
:align: center
```
This pop up will appear
``` {image} images/tunnel_demo2.png
:alt: MobaXterm Pop Up
:class: bg-primary
:align: center
```
3. Press `New SSH tunnel`
``` {image} images/tunnel_demo3.png
:alt: MobaXterm Pop Up
:class: bg-primary
:align: center
```
4. Configure tunnel accordingly

   Local clients
   - Forwarded Port: [PORT]

   Remote Server
   - Remote Server: localhost
   - Remote Port: [PORT]

   SSH Server
   - SSH server: Name of the server one is connecting to
   - SSH login: Username to login to the server
   - SSH port: 22
