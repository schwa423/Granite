#include "render_graph.hpp"
#include <algorithm>

using namespace std;

namespace Granite
{
RenderTextureResource &RenderPass::add_attachment_input(const std::string &name)
{
	auto &res = graph.get_texture_resource(name);
	res.read_in_pass(index);
	attachments_inputs.push_back(&res);
	return res;
}

RenderTextureResource &RenderPass::add_color_input(const std::string &name)
{
	auto &res = graph.get_texture_resource(name);
	res.read_in_pass(index);
	color_inputs.push_back(&res);
	color_scale_inputs.push_back(nullptr);
	return res;
}

RenderTextureResource &RenderPass::add_texture_input(const std::string &name)
{
	auto &res = graph.get_texture_resource(name);
	res.read_in_pass(index);
	texture_inputs.push_back(&res);
	return res;
}

RenderTextureResource &RenderPass::add_color_output(const std::string &name, const AttachmentInfo &info)
{
	auto &res = graph.get_texture_resource(name);
	res.written_in_pass(index);
	res.set_attachment_info(info);
	color_outputs.push_back(&res);
	return res;
}

RenderTextureResource &RenderPass::set_depth_stencil_output(const std::string &name, const AttachmentInfo &info)
{
	auto &res = graph.get_texture_resource(name);
	res.written_in_pass(index);
	res.set_attachment_info(info);
	depth_stencil_output = &res;
	return res;
}

RenderTextureResource &RenderPass::set_depth_stencil_input(const std::string &name)
{
	auto &res = graph.get_texture_resource(name);
	res.read_in_pass(index);
	depth_stencil_input = &res;
	return res;
}

RenderTextureResource &RenderGraph::get_texture_resource(const std::string &name)
{
	auto itr = resource_to_index.find(name);
	if (itr != end(resource_to_index))
	{
		return static_cast<RenderTextureResource &>(*resources[itr->second]);
	}
	else
	{
		unsigned index = resources.size();
		resources.emplace_back(new RenderTextureResource(index));
		resource_to_index[name] = index;
		return static_cast<RenderTextureResource &>(*resources.back());
	}
}

RenderPass &RenderGraph::add_pass(const std::string &name)
{
	auto itr = pass_to_index.find(name);
	if (itr != end(pass_to_index))
	{
		return *passes[itr->second];
	}
	else
	{
		unsigned index = passes.size();
		passes.emplace_back(new RenderPass(*this, index));
		pass_to_index[name] = index;
		return *passes.back();
	}
}

void RenderGraph::set_backbuffer_source(const std::string &name)
{
	backbuffer_source = name;
}

void RenderGraph::validate_passes()
{
	for (auto &pass_ptr : passes)
	{
		auto &pass = *pass_ptr;
		if (!pass.get_color_inputs().empty() && pass.get_color_inputs().size() != pass.get_color_outputs().size())
			throw logic_error("Size of color inputs must match color outputs.");

		if (!pass.get_color_inputs().empty())
		{
			unsigned num_inputs = pass.get_color_inputs().size();
			for (unsigned i = 0; i < num_inputs; i++)
			{
				if (get_resource_dimensions(*pass.get_color_inputs()[i]) != get_resource_dimensions(*pass.get_color_outputs()[i]))
					pass.make_color_input_scaled(i);
			}
		}

		if (pass.get_depth_stencil_input() && pass.get_depth_stencil_output())
		{
			if (get_resource_dimensions(*pass.get_depth_stencil_input()) != get_resource_dimensions(*pass.get_depth_stencil_output()))
				throw logic_error("Dimension mismatch.");
		}
	}
}

void RenderGraph::build_physical_passes()
{
	physical_passes.clear();
	PhysicalPass physical_pass;

	const auto should_merge = [](const RenderPass &start, const RenderPass &next) -> bool {
		return false;
	};

	for (unsigned index = 0; index < pass_stack.size(); )
	{
		unsigned merge_end = index + 1;
		for (; merge_end < pass_stack.size(); merge_end++)
		{
			if (!should_merge(*passes[pass_stack[index]], *passes[pass_stack[merge_end]]))
				break;
		}

		physical_pass.passes.insert(end(physical_pass.passes), begin(pass_stack) + index, begin(pass_stack) + merge_end);
		physical_passes.push_back(move(physical_pass));
		index = merge_end;
	}
}

void RenderGraph::bake()
{
	validate_passes();
	build_physical_passes();

	auto itr = resource_to_index.find(backbuffer_source);
	if (itr == end(resource_to_index))
		throw logic_error("Backbuffer source does not exist.");

	pushed_passes.clear();
	pushed_passes_tmp.clear();
	pass_stack.clear();
	handled_passes.clear();

	auto &backbuffer_resource = *resources[itr->second];

	if (backbuffer_resource.get_write_passes().empty())
		throw logic_error("No pass exists which writes to resource.");

	for (auto &pass : backbuffer_resource.get_write_passes())
	{
		pass_stack.push_back(pass);
		pushed_passes.push_back(pass);
	}

	const auto depend_passes = [&](const std::unordered_set<unsigned> &passes) {
		if (passes.empty())
			throw logic_error("No pass exists which writes to resource.");

		for (auto &pass : passes)
		{
			pushed_passes_tmp.push_back(pass);
			pass_stack.push_back(pass);
		}
	};

	const auto make_unique_list = [](std::vector<unsigned> &passes) {
		sort(begin(passes), end(passes));
		passes.erase(unique(begin(passes), end(passes)), end(passes));
	};

	unsigned iteration_count = 0;

	while (!pushed_passes.empty())
	{
		pushed_passes_tmp.clear();
		make_unique_list(pushed_passes);

		for (auto &pushed_pass : pushed_passes)
		{
			handled_passes.insert(pushed_pass);

			auto &pass = *passes[pushed_pass];
			if (pass.get_depth_stencil_input())
				depend_passes(pass.get_depth_stencil_input()->get_write_passes());
			for (auto *input : pass.get_attachment_inputs())
				depend_passes(input->get_write_passes());
			for (auto *input : pass.get_color_inputs())
				if (input)
					depend_passes(input->get_write_passes());
			for (auto *input : pass.get_color_scale_inputs())
				if (input)
					depend_passes(input->get_write_passes());
			for (auto *input : pass.get_texture_inputs())
				depend_passes(input->get_write_passes());
		}

		pushed_passes.clear();
		swap(pushed_passes, pushed_passes_tmp);

		if (++iteration_count > passes.size())
			throw logic_error("Cycle detected.");
	}

	reverse(begin(pass_stack), end(pass_stack));
	filter_passes(pass_stack);

	pass_barriers.clear();
	pass_barriers.reserve(pass_stack.size());

	build_barriers();
}

ResourceDimensions RenderGraph::get_resource_dimensions(const RenderTextureResource &resource) const
{
	ResourceDimensions dim;
	auto &info = resource.get_attachment_info();
	dim.format = info.format;

	switch (info.size_class)
	{
	case SizeClass::SwapchainRelative:
		dim.width = unsigned(info.size_x * swapchain_dimensions.width);
		dim.height = unsigned(info.size_y * swapchain_dimensions.height);
		break;

	case SizeClass::Absolute:
		dim.width = unsigned(info.size_x);
		dim.height = unsigned(info.size_y);
		break;

	case SizeClass::InputRelative:
	{
		auto itr = resource_to_index.find(info.size_relative_name);
		if (itr == end(resource_to_index))
			throw logic_error("Resource does not exist.");
		auto &input = static_cast<RenderTextureResource &>(*resources[itr->second]);
		auto input_dim = get_resource_dimensions(input);

		dim.width = unsigned(input_dim.width * info.size_x);
		dim.height = unsigned(input_dim.height * info.size_y);
		dim.depth = input_dim.depth;
		dim.layers = input_dim.layers;
		dim.levels = input_dim.levels;
		break;
	}
	}

	if (dim.format == VK_FORMAT_UNDEFINED)
		dim.format = swapchain_dimensions.format;

	return dim;
}

void RenderGraph::build_barriers()
{
	const auto get_access = [&](vector<Barrier> &barriers, unsigned index) -> Barrier & {
		auto itr = find_if(begin(barriers), end(barriers), [index](const Barrier &b) {
			return index == b.resource_index;
		});
		if (itr != end(barriers))
			return *itr;
		else
		{
			barriers.push_back({ index, VK_IMAGE_LAYOUT_UNDEFINED, 0 });
			return barriers.back();
		}
	};

	for (auto &index : pass_stack)
	{
		auto &pass = *passes[index];
		Barriers barriers;

		const auto get_dst_access = [&](unsigned index) -> Barrier & {
			return get_access(barriers.dst_access, index);
		};

		const auto get_src_access = [&](unsigned index) -> Barrier & {
			return get_access(barriers.src_access, index);
		};

		for (auto *input : pass.get_texture_inputs())
		{
			auto &barrier = get_dst_access(input->get_index());
			barrier.access |= VK_ACCESS_SHADER_READ_BIT;
			if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
				throw logic_error("Layout mismatch.");
			barrier.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		for (auto *input : pass.get_attachment_inputs())
		{
			auto &barrier = get_dst_access(input->get_index());
			barrier.access |= VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
			if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
				throw logic_error("Layout mismatch.");
			barrier.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		for (auto *input : pass.get_color_inputs())
		{
			if (!input)
				continue;

			auto &barrier = get_dst_access(input->get_index());
			barrier.access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
			if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
				throw logic_error("Layout mismatch.");
			barrier.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		}

		for (auto *input : pass.get_color_scale_inputs())
		{
			if (!input)
				continue;

			auto &barrier = get_dst_access(input->get_index());
			barrier.access |= VK_ACCESS_SHADER_READ_BIT;
			if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
				throw logic_error("Layout mismatch.");
			barrier.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		for (auto *output : pass.get_color_outputs())
		{
			auto &barrier = get_src_access(output->get_index());
			barrier.access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			if (barrier.layout != VK_IMAGE_LAYOUT_UNDEFINED)
				throw logic_error("Layout mismatch.");
			barrier.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		}

		auto *output = pass.get_depth_stencil_output();
		auto *input = pass.get_depth_stencil_input();

		if (output && input)
		{
			auto &dst_barrier = get_dst_access(input->get_index());
			auto &src_barrier = get_src_access(output->get_index());

			if (dst_barrier.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
				dst_barrier.layout = VK_IMAGE_LAYOUT_GENERAL;
			else
				dst_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			dst_barrier.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			src_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			src_barrier.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		}
		else if (input)
		{
			auto &dst_barrier = get_dst_access(input->get_index());
			if (dst_barrier.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
				dst_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
			else
				dst_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			dst_barrier.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		}
		else if (output)
		{
			auto &src_barrier = get_src_access(output->get_index());
			src_barrier.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			src_barrier.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		}

		pass_barriers.push_back(move(barriers));
	}
}

void RenderGraph::filter_passes(std::vector<unsigned> &list)
{
	unordered_set<unsigned> seen;

	auto output_itr = begin(list);
	for (auto itr = begin(list); itr != end(list); ++itr)
	{
		if (!seen.count(*itr))
		{
			*output_itr = *itr;
			seen.insert(*itr);
			++output_itr;
		}
	}
	list.erase(output_itr, end(list));
}

void RenderGraph::reset()
{
	passes.clear();
	resources.clear();
	pass_to_index.clear();
	resource_to_index.clear();
	physical_passes.clear();
}

}