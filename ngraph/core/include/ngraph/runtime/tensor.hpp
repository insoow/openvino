//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <vector>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        class NGRAPH_API Tensor
        {
        protected:
            Tensor(const std::shared_ptr<ngraph::descriptor::Tensor>& descriptor)
                : m_descriptor(descriptor)
                , m_stale(true)
            {
            }

        public:
            virtual ~Tensor() {}
            Tensor& operator=(const Tensor&) = default;

            /// \brief Get tensor shape
            /// \return const reference to a Shape
            virtual const ngraph::Shape& get_shape() const;

            /// \brief Get tensor partial shape
            /// \return const reference to a PartialShape
            const ngraph::PartialShape& get_partial_shape() const;

            /// \brief Get tensor element type
            /// \return element::Type
            virtual const element::Type& get_element_type() const;

            /// \brief Get number of elements in the tensor
            /// \return number of elements in the tensor
            virtual size_t get_element_count() const;

            /// \brief Get the size in bytes of the tensor
            /// \return number of bytes in tensor's allocation
            virtual size_t get_size_in_bytes() const;

            /// \brief Get tensor's unique name
            /// \return tensor's name
            const std::string& get_name() const;

            /// \brief Get the stale value of the tensor. A tensor is stale if its data is
            /// changed.
            /// \return true if there is new data in this tensor
            bool get_stale() const;

            /// \brief Set the stale value of the tensor. A tensor is stale if its data is
            /// changed.
            void set_stale(bool val);

            /// \brief Write bytes directly into the tensor
            /// \param p Pointer to source of data
            /// \param n Number of bytes to write, must be integral number of elements.
            virtual void write(const void* p, size_t n) = 0;

            /// \brief Read bytes directly from the tensor
            /// \param p Pointer to destination for data
            /// \param n Number of bytes to read, must be integral number of elements.
            virtual void read(void* p, size_t n) const = 0;

            /// \brief check tensor for new data, call may block.
            ///    backends may use this to ensure tensor is updated (eg: lazy eval).
            virtual void wait_for_read_ready() {}
            /// \brief notify tensor of new data, call may block.
            ///    backends may use this as indication of new data in tensor.
            virtual void wait_for_write_ready() {}
        protected:
            std::shared_ptr<ngraph::descriptor::Tensor> m_descriptor;
            bool m_stale;
        };
    }
}
