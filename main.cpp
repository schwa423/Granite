#include "util.hpp"
#include "compiler.hpp"
#include "filesystem.hpp"
#include "event.hpp"
#include "path.hpp"
#include "render_queue.hpp"
#include <unistd.h>
#include <string.h>
#include <wsi/wsi.hpp>

using namespace Granite;
using namespace std;

int main()
{
	Filesystem::get();
	EventManager manager;

	manager.enqueue<AEvent>(10);

	class Handler : public EventHandler
	{
	public:
		bool handle_a(const Event &e)
		{
			auto &event = e.as<AEvent>();
			fprintf(stderr, "%d\n", event.a);
			return false;
		}

		bool handle_a2(const Event &e)
		{
			auto &event = e.as<AEvent>();
			fprintf(stderr, "%d\n", event.a + 1);
			return true;
		}

		void up(const Event &e)
		{
			fprintf(stderr, "UP %d\n", e.as<BEvent>().b);
		}

		void down(const Event &e)
		{
			fprintf(stderr, "DOWN %d\n", e.as<BEvent>().b);
		}

		void up2(const Event &e)
		{
			fprintf(stderr, "UP2 %d\n", e.as<BEvent>().b);
		}

		void down2(const Event &e)
		{
			fprintf(stderr, "DOWN2 %d\n", e.as<BEvent>().b);
		}
	} handler;

	//manager.register_handler(AEvent::type_id, static_cast<bool (EventHandler::*)(const Event &event)>(&Handler::handle_a), &handler);
	manager.register_handler(AEvent::type_id, &Handler::handle_a, &handler);
	manager.register_handler(AEvent::type_id, &Handler::handle_a2, &handler);
	manager.dispatch();

	manager.enqueue<AEvent>(20);
	manager.unregister_handler(&handler);
	manager.dispatch();

	manager.register_latch_handler(BEvent::type_id, &Handler::up, &Handler::down, &handler);
	auto cookie = manager.enqueue_latched<BEvent>(10, 20);
	manager.dequeue_latched(cookie);
	manager.register_latch_handler(BEvent::type_id, &Handler::up2, &Handler::down2, &handler);

	cookie = manager.enqueue_latched<BEvent>(10, 40);

	try {
	Vulkan::WSI wsi;
	if (!wsi.init(1280, 720))
		return 1;

    auto &device = wsi.get_device();

    GLSLCompiler compiler;
    compiler.set_source_from_file("assets://shaders/quad.vert");
	compiler.preprocess();
	auto vert = compiler.compile();
    if (vert.empty())
        LOGE("Error: %s\n", compiler.get_error_message().c_str());

    compiler.set_source_from_file("assets://shaders/quad.frag");
	compiler.preprocess();
	auto frag = compiler.compile();
    if (frag.empty())
        LOGE("Error: %s\n", compiler.get_error_message().c_str());

    compiler.set_source_from_file("assets://shaders/depth.frag");
	compiler.preprocess();
	auto depth_frag = compiler.compile();
	if (depth_frag.empty())
		LOGE("Error: %s\n", compiler.get_error_message().c_str());

	auto program = device.create_program(vert.data(), vert.size() * sizeof(uint32_t), frag.data(), frag.size() * sizeof(uint32_t));
	auto depth_program = device.create_program(vert.data(), vert.size() * sizeof(uint32_t), depth_frag.data(), depth_frag.size() * sizeof(uint32_t));

	while (wsi.alive())
	{
		wsi.begin_frame();

		auto cmd = device.request_command_buffer();
		auto rp = device.get_swapchain_render_pass(Vulkan::SwapchainRenderPass::DepthStencil);
		rp.clear_color[0].float32[0] = 1.0f;
		rp.clear_color[0].float32[1] = 0.5f;
		rp.clear_color[0].float32[2] = 0.5f;
		rp.clear_color[0].float32[3] = 1.0f;
		rp.clear_depth_stencil.depth = 0.2f;
		rp.clear_depth_stencil.stencil = 128;

		Vulkan::RenderPassInfo::Subpass subpasses[2] = {};
		subpasses[0].num_color_attachments = 1;
		subpasses[0].color_attachments[0] = 0;
		subpasses[0].depth_stencil_mode = Vulkan::RenderPassInfo::DepthStencil::ReadWrite;
		subpasses[1].num_color_attachments = 1;
		subpasses[1].color_attachments[0] = 0;
		subpasses[1].num_input_attachments = 1;
		subpasses[1].input_attachments[0] = 1;
		subpasses[1].depth_stencil_mode = Vulkan::RenderPassInfo::DepthStencil::ReadOnly;
		rp.num_subpasses = 2;
		rp.subpasses = subpasses;

		cmd->begin_render_pass(rp);

		cmd->set_program(*program);
        static const int8_t quad[] = {
				-128, -128,
				+127, -128,
				-128, +127,
                +127, +127,
		};
		memcpy(cmd->allocate_vertex_data(0, 6, 2), quad, sizeof(quad));
		cmd->set_quad_state();
		cmd->set_depth_test(true, true);
		cmd->set_depth_compare(VK_COMPARE_OP_LESS_OR_EQUAL);
		cmd->set_vertex_attrib(0, 0, VK_FORMAT_R8G8_SNORM, 0);
		cmd->set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
		cmd->draw(3);
		cmd->next_subpass();

		cmd->set_program(*depth_program);
		memcpy(cmd->allocate_vertex_data(0, 8, 2), quad, sizeof(quad));
		cmd->set_quad_state();
		cmd->set_vertex_attrib(0, 0, VK_FORMAT_R8G8_SNORM, 0);
		cmd->set_primitive_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
		cmd->set_input_attachments(0, 0);
		cmd->set_input_attachments(0, 1);
		cmd->draw(4);

		cmd->end_render_pass();

		device.submit(cmd);
		wsi.end_frame();
	}
	}
    catch (const exception &e)
	{
		LOGE("Exception: %s\n", e.what());
	}

#if 0
	GLSLCompiler compiler;
	compiler.set_source_from_file(Filesystem::get(), "/tmp/test.frag");
	compiler.preprocess();
	auto spirv = compiler.compile();

	if (spirv.empty())
		LOGE("GLSL: %s\n", compiler.get_error_message().c_str());

	for (auto &dep : compiler.get_dependencies())
		LOGI("Dependency: %s\n", dep.c_str());
	for (auto &dep : compiler.get_variants())
		LOGI("Variant: %s\n", dep.first.c_str());

	auto &fs = Filesystem::get();
	auto file = fs.open("/tmp/foobar", Filesystem::Mode::WriteOnly);
	const string foo = ":D";
	memcpy(file->map_write(foo.size()), foo.data(), foo.size());
#endif
}
