




def learn_rspmn(self, data, total_num_of_time_steps_varies):
    self.template = copy.deepcopy(self.InitialTemplate.template_network)

    node_functions = get_node_funtions()
    _node_functions_top_down = node_functions[0].copy()
    # _node_functions_top_down.update({Sum: self.mpe_interface_sum})
    logging.debug(f'_node_functions_top_down {_node_functions_top_down}')

    if total_num_of_time_steps_varies:
        assert type(data) is list, 'When sequence length varies, data is ' \
                                   'a list of numpy arrays'
        print("Evaluating rspn and collecting nodes to update")

        for row in range(len(data)):
            each_data_point = data[row]
            # print("length of sequence:", self.get_len_sequence())

            unrolled_network_lls_per_node = \
                self.eval_rspmn_bottom_up(
                    each_data_point
                )
    else:
        assert type(data) is np.ndarray, 'data should be of type numpy' \
                                         ' array'

        # print("Length of the sequence in mini_batch:", len_seq)
        # assert data.shape[
        #            1] == self.num_variables * self.len_sequence,
        #            "data columns not equal to number of variables
        #            time length of sequence"
        # # print("Evaluating rspn and collecting nodes to update")

        unrolled_network_lls_per_node = self.eval_rspmn_bottom_up(
            self.template, data, True
        )
        self.eval_rspmn_top_down(
            self.template, data, unrolled_network_lls_per_node,
            _node_functions_top_down
        )

    self.update_weights(self.template)


def eval_rspmn_bottom_up(self, template, data, *args):
    # assert self.InitialTemplate.top_network is not None,
    # f'top layer does not exist'
    # assert self.template is not None, f'template layer does not exist'

    assert type(data) is np.ndarray, 'data should be of type numpy array'

    num_variables_each_time_step, total_num_of_time_steps, \
    initial_num_latent_interface_nodes = \
        self.get_params_for_get_each_time_step_data_for_template(template,
                                                                 data)

    logging.debug(
        f'intial_num_latent_interface_nodes '
        f'{initial_num_latent_interface_nodes}')
    logging.debug(f'total_num_of_time_steps {total_num_of_time_steps}')

    template_nodes = get_nodes_by_type(template)

    # for bottom most time step + 1
    eval_val_per_node = np.zeros((data.shape[0], len(template_nodes)))
    unrolled_network_eval_val_per_node = [eval_val_per_node]

    # evaluate template bottom up at each time step
    for time_step_num_in_reverse_order in range(total_num_of_time_steps - 1,
                                                -1, -1):

        logging.debug(
            f'time_step_num_in_reverse_order '
            f'{time_step_num_in_reverse_order}')

        prev_eval_val_per_node = unrolled_network_eval_val_per_node[-1]
        logging.debug(f'prev_eval_val_per_node {prev_eval_val_per_node}')

        each_time_step_data_for_template = \
            self.get_each_time_step_data_for_template(
                data,
                time_step_num_in_reverse_order,
                total_num_of_time_steps,
                prev_eval_val_per_node,
                initial_num_latent_interface_nodes,
                num_variables_each_time_step,
                bottom_up=True
            )

        if time_step_num_in_reverse_order == 0:

            top_nodes = get_nodes_by_type(self.InitialTemplate.top_network)
            eval_val_per_node = np.zeros((data.shape[0], len(top_nodes)))
            if args[0]:
                log_likelihood(self.InitialTemplate.top_network,
                               each_time_step_data_for_template,
                               lls_matrix=eval_val_per_node)

            else:
                result, meu_matrix = meu(self.InitialTemplate.top_network,
                                         each_time_step_data_for_template,
                                         meu_matrix=eval_val_per_node)

                eval_val_per_node = meu_matrix
                print(f'eval_val_per_node {eval_val_per_node}')
                # print(f'meu_matrix {meu_matrix}')

        else:
            eval_val_per_node = np.zeros((data.shape[0], len(template_nodes)))
            if args[0]:
                log_likelihood(template, each_time_step_data_for_template,
                               lls_matrix=eval_val_per_node)

            else:

                self.pass_meu_val_to_latent_interface_leaf_nodes(
                    eval_val_per_node, prev_eval_val_per_node,
                    initial_num_latent_interface_nodes)

                result, meu_matrix = meu(template,
                                         each_time_step_data_for_template,
                                         meu_matrix=eval_val_per_node)

                eval_val_per_node = meu_matrix
                print(f'eval_val_per_node {eval_val_per_node}')

        unrolled_network_eval_val_per_node.append(eval_val_per_node)

    # print(np.mean(unrolled_network_eval_val_per_node[-1][:, 0]))

    return unrolled_network_eval_val_per_node


def eval_rspmn_top_down(self, template, data,
                        unrolled_network_lls_per_node,
                        node_functions_top_down=None
                        ):
    logging.debug(f'in method eval_rspmn_top_down()')

    num_variables_each_time_step, total_num_of_time_steps, \
    initial_num_latent_interface_nodes = \
        self.get_params_for_get_each_time_step_data_for_template(template,
                                                                 data)

    for time_step_num in range(total_num_of_time_steps - 1):
        lls_per_node = unrolled_network_lls_per_node[
            total_num_of_time_steps - time_step_num
            ]

        each_time_step_data_for_template = \
            self.get_each_time_step_data_for_template(
                data, time_step_num,
                total_num_of_time_steps,
                lls_per_node,
                initial_num_latent_interface_nodes,
                num_variables_each_time_step,
                bottom_up=False
            )

        instance_ids = np.arange(each_time_step_data_for_template.shape[0])
        if time_step_num == 0:
            all_results, latent_interface_dict = eval_template_top_down(
                self.InitialTemplate.top_network,
                node_functions_top_down, False,
                all_results=None, parent_result=instance_ids,
                data=each_time_step_data_for_template,
                lls_per_node=lls_per_node)

        else:

            all_results, latent_interface_dict = eval_template_top_down(
                template,
                node_functions_top_down, False,
                all_results=None, parent_result=instance_ids,
                data=each_time_step_data_for_template,
                lls_per_node=lls_per_node
            )
        template.interface_winner = np.full(
            (each_time_step_data_for_template.shape[0],), np.inf
        )
        logging.debug(f'latent_interface_dict {latent_interface_dict}')
        for latent_interface_node, instances in \
                latent_interface_dict.items():
            template.interface_winner[instances] = \
                latent_interface_node.interface_idx - \
                num_variables_each_time_step

        # if self.template.interface_winner.any(np.inf):
        #     raise Exception(f'All instances are not passed to
        #     the corresponding latent interface nodes')
