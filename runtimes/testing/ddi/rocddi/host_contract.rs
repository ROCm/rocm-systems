// SPDX-License-Identifier: MIT

//! Native host allocation lifetime through the production frontend-facing DDI.

use rocddi::memory::host_page_size;
use rocddi::session::{Session, SessionLifetime};
use std::error::Error;

fn require(condition: bool, detail: &str) -> Result<(), Box<dyn Error>> {
    if condition {
        Ok(())
    } else {
        Err(std::io::Error::other(detail).into())
    }
}

#[allow(unsafe_code)]
fn host_allocation_contract() -> Result<(), Box<dyn Error>> {
    let page = host_page_size()?;
    for lifetime in [SessionLifetime::Process, SessionLifetime::Session] {
        let mut session = Session::new(lifetime)?;
        require(
            session.allocate_host(0, page).is_err(),
            "zero extent accepted",
        )?;
        let mut allocation = session.allocate_host(page * 2, page * 16)?;
        let info = allocation.info();
        require(
            info.host_address != 0,
            "allocation returned a null host address",
        )?;
        require(info.size == page * 2, "logical extent changed")?;
        require(
            info.host_address as u64 % (page * 16) == 0,
            "host address is not aligned",
        )?;
        // Host storage owns no session connection: destroying the controller
        // must leave this allocation writable until its explicit free.
        session.destroy()?;
        require(
            session.allocate_host(page, page).is_err(),
            "destroyed session accepted an allocation",
        )?;
        {
            // SAFETY: The production allocation owns this nonempty mapped range.
            // This sole mutable borrow ends before free and never escapes.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    info.host_address as *mut u8,
                    usize::try_from(info.size)?,
                )
            };
            bytes.fill(0x5a);
            require(
                bytes.iter().all(|byte| *byte == 0x5a),
                "host storage mismatch",
            )?;
        }
        allocation.free()?;
        allocation.free()?;
        require(allocation.info() == info, "cleanup changed cached facts")?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    host_allocation_contract()?;
    println!("PASS ddi.rocddi.host-allocation");
    Ok(())
}
