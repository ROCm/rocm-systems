/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// @brief Key class for implementing the PassKey idiom
///
/// If you make this a parameter of a function, that function will be
/// private to every class aside from the one specified in the template
/// type. This allows friend semantics for a limited number of functions
/// instead of the entire class.
///
/// This is used in hipFile to make the constructors in File, Buffer,
/// etc. private to everyone aside from their associated container classes,
/// which makes it impossible to instantiate an unregistered object.
///
/// @tparam T The "friend" class
template <typename T> class PassKey {
    friend T;

private:
    // Note: Changing these constructors can break encapsulation.

    PassKey()
    {
    }

    PassKey(PassKey const &)
    {
    }
};
