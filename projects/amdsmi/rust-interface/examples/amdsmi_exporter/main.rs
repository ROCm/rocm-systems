// Copyright (C) 2024 Advanced Micro Devices. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#[macro_use]
mod amdsmi_collectors;
#[macro_use]
mod amdsmi_metric;
mod collectors;
mod http_server;

use crate::amdsmi_collectors::*;
use clap::{App, Arg, SubCommand};
use http_server::run_http_server;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

const APP_VERSION_STR: &str = "1.0";

#[tokio::main]
async fn main() {
    // Parse command line arguments
    let cli_matches = App::new("AMDSMI Exporter")
        .version(APP_VERSION_STR)
        .author("AMD <amd-smi.support@amd.com>")
        .about("Exports AMD SMI metrics")
        .subcommand(
            SubCommand::with_name("list")
                .about("Lists the supported GPUs and metrics' information")
        )
        .subcommand(
            SubCommand::with_name("metric")
                .about("Exports AMD SMI metrics and runs the HTTP server")
                .arg(
                    Arg::new("port")
                        .short('p')
                        .long("port")
                        .value_name("PORT")
                        .help("Sets the port to listen on")
                        .takes_value(true),
                )
                .arg(
                    Arg::new("gpu")
                        .short('g')
                        .long("gpu")
                        .value_name("GPU")
                        .help("Sets the GPU BDF or the zero-based index of the socket ID and processor ID for export. Example: --gpu 0000:03:00.0 or --gpu 0.0. You can specify multiple GPUs separated by commas")
                        .takes_value(true)
                )
                .arg(
                    Arg::new("whitelist")
                        .short('w')
                        .long("whitelist")
                        .value_name("METRIC")
                        .help("Comma-separated list of metric field names to include")
                        .takes_value(true),
                )
                .arg(
                    Arg::new("blacklist")
                        .short('b')
                        .long("blacklist")
                        .value_name("METRIC")
                        .help("Comma-separated list of metric field names to exclude")
                        .takes_value(true),
                )
        )
        .get_matches();

    let collectors = Arc::new(Mutex::new(AmdsmiCollectors::default()));

    match cli_matches.subcommand() {
        Some(("list", _)) => {
            collectors.lock().await.print_supported();
        }
        Some(("metric", arg_options)) => {
            export_metrics(arg_options, &collectors).await;
        }
        _ => {
            eprintln!("Invalid command. Please use 'list' to list supported GPUs and metrics, or 'metric' to export metrics and run the HTTP server. Use --help for more information.");
        }
    }
}

async fn export_metrics(arg_options: &clap::ArgMatches, collectors: &Arc<Mutex<AmdsmiCollectors>>) {
    let port: u16 = arg_options
        .get_one::<String>("port")
        .unwrap_or(&"9898".to_string())
        .parse::<u16>()
        .unwrap_or_else(|_| {
            eprintln!("Invalid port number provided. Using default port 9898.");
            9898
        });

    let gpu_filters: Option<Vec<String>> = arg_options
        .get_one::<String>("gpu")
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());

    let whitelist: Option<Vec<String>> = arg_options
        .get_one::<String>("whitelist")
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());

    let blacklist: Option<Vec<String>> = arg_options
        .get_one::<String>("blacklist")
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());

    collectors
        .lock()
        .await
        .set_gpu_filter(gpu_filters)
        .set_field_filter(whitelist, blacklist);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    // Run the HTTP server
    run_http_server(&collectors, addr).await;
}
