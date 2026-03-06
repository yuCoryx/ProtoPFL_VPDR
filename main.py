from utils.init import Node, init_model, init_optimizer, create_client_heterogeneous_model
from server import Server_update
from client import Client_update, Client_encode
from utils.utils import setup_seed, get_model_info, validate, validate_fedpcl
from utils.dp_utils import calibrate_sigma_prototype_rdp
from utils.label_skew import get_federated_loaders as get_label_fed_loaders
from utils.domain_skew import get_federated_loaders as get_domain_fed_loaders
from options import args_parser
from attacks.hijack import run_hijack_eval

import numpy as np
import os, time, logging, json, copy, warnings

warnings.filterwarnings("ignore", category=Warning)


def main():
    args = args_parser()
    setup_seed(args.random_seed)

    # Logging setup
    time_stamp = time.strftime('%Y%m%d_%H%M%S')
    args.output_dir = f'logs/{args.exp_name}/{args.dataset}_alpha{args.dirichlet_alpha}/N{args.node_num}_T{args.T}_E{args.E}'
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, f'{args.method}_{time_stamp}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()], force=True)
    logging.info('Starting federated training')

    all_avg_accs = []

    # Federated data partitioning
    if args.dataset in ('office_caltech10', 'pacs', 'digits'):
        train_loaders, test_loaders, num_classes = get_label_fed_loaders(dataset=args.dataset, data_root=args.data_root,
                                                                         num_clients=args.node_num, batch_size=args.batch_size,
                                                                         alpha=args.dirichlet_alpha, num_workers=args.num_workers)
    else:
        train_loaders, test_loaders, num_classes = get_domain_fed_loaders(dataset=args.dataset, data_root=args.data_root,
                                                                          batch_size=args.batch_size, num_workers=args.num_workers)
        args.node_num = len(train_loaders)
        args.dirichlet_alpha = 0.0

    args.sample_sizes = [len(loader.dataset) for loader in train_loaders.values()]
    args.num_classes = num_classes
    logging.info(f'Loaded data for {len(train_loaders)} clients')

    # DP calibration (prototype release via RDP)
    if args.privacy == 'dp': 
        logging.info("=" * 80) 

        if args.noise_add == 'vpp':
            args.vpp_topk_eps = args.epsilon * args.vpp_topk_eps_ratio
            epsilon_for_proto = args.epsilon - args.vpp_topk_eps
            logging.info(f"[DP] Budget split: eps_topk={args.vpp_topk_eps}, eps_proto={epsilon_for_proto}, eps_total={args.epsilon}")
        else:
            epsilon_for_proto = args.epsilon
            logging.info(f"[DP] Target budget: (eps={args.epsilon}, delta={args.delta})")

        args.noise_multiplier = calibrate_sigma_prototype_rdp(num_releases=args.T, target_epsilon=epsilon_for_proto, target_delta=args.delta)
        logging.info(f"[DP] Calibrated noise multiplier: sigma={args.noise_multiplier:.4f}")
        logging.info("=" * 80)

    else:
        args.noise_multiplier = 0.0

    # Model initialization (homogeneous or heterogeneous)
    if args.enable_heterogeneous:
        logging.info(f'Heterogeneous mode enabled: family={args.model_family}, feature_dim={args.feature_dim}')

        base = init_model(args, num_classes).to(args.device)
        server_model = copy.deepcopy(base)
        weight_decay = getattr(args, 'weight_decay', 5e-4)
        server_opt = init_optimizer(server_model, args.optimizer, args.lr, weight_decay)
        server = Node(-1, None, test_loaders.get(-1, next(iter(test_loaders.values()))), num_classes, server_model, server_opt, args)

        clients = {}
        for cid in range(args.node_num):
            client_model = create_client_heterogeneous_model(cid, args, num_classes)
            if client_model is None:
                logging.warning(f'Client {cid}: heterogeneous model creation failed; fallback to homogeneous.')
                client_model = init_model(args, num_classes)
            else:
                logging.info(f'Client {cid} model: {get_model_info(client_model)}')

            client_model = client_model.to(args.device)
            opt = init_optimizer(client_model, args.optimizer, args.lr, weight_decay)
            clients[cid] = Node(cid, train_loaders[cid], test_loaders[cid], num_classes, client_model, opt, args)

    else:
        logging.info('Homogeneous model mode')

        base = init_model(args, num_classes).to(args.device)
        server_model = copy.deepcopy(base)
        weight_decay = getattr(args, 'weight_decay', 5e-4)
        server_opt = init_optimizer(server_model, args.optimizer, args.lr, weight_decay)
        server = Node(-1, None, test_loaders.get(-1, next(iter(test_loaders.values()))), num_classes, server_model, server_opt, args)

        clients = {}
        for cid in range(args.node_num):
            m = copy.deepcopy(base)
            opt = init_optimizer(m, args.optimizer, args.lr, weight_decay)
            clients[cid] = Node(cid, train_loaders[cid], test_loaders[cid], num_classes, m, opt, args)

    logging.info('Initialized server and client nodes')
    logging.info(args)

    t_global_start = time.time()
    all_hijack_results = [] if getattr(args, "eval_hijack", False) else None
    all_mia_results = [] if getattr(args, "eval_mia", False) else None

    # Training loop
    for rnd in range(args.T):
        logging.info("Round %d/%d", rnd + 1, args.T)

        # Optional per-round LR decay
        if hasattr(args, 'lr_decay') and args.lr_decay < 1.0 and rnd > 0:
            lr_decay_factor = args.lr_decay ** rnd
            current_lr = args.lr * lr_decay_factor
            for client in clients.values():
                for pg in client.optimizer.param_groups:
                    pg['lr'] = current_lr
            for pg in server.optimizer.param_groups:
                pg['lr'] = current_lr
            logging.info(f"LR decayed to {current_lr:.6f} (factor={lr_decay_factor:.4f})")

        select_list = list(clients.keys())
        logging.info(f"Selected clients: {select_list}")

        if args.method in ['fedproto', 'fedplvm', 'fpl', 'fedpcl', 'mpft', 'fedtgp']:
            logging.info('=== Client encode ===')
            args.current_round = rnd + 1
            Client_encode(args, clients, select_list, round_idx=rnd + 1)

            # Prototype hijack attack (before aggregation)
            if getattr(args, "eval_hijack", False):
                from attacks.hijack import run_hijack_eval_multi_clients
                logging.info('=== Hijack attack (before aggregation) ===')
                round_result = run_hijack_eval_multi_clients(
                    args, server, clients, select_list, rnd,
                    per_class=getattr(args, "hijack_per_class", 1),
                    max_classes=getattr(args, "hijack_max_classes", 3),
                    steps=getattr(args, "hijack_steps", 10000),
                    lr=getattr(args, "hijack_lr", 0.01),
                    tv_weight=getattr(args, "hijack_tv", 1e-3),
                    l2_weight=getattr(args, "hijack_l2", 0.0),
                    aug_consistency=getattr(args, "hijack_aug", False),
                    early_stop_patience=getattr(args, "hijack_early_stop_patience", 500),
                    early_stop_threshold=getattr(args, "hijack_early_stop_threshold", 1e-6),
                    min_steps=getattr(args, "hijack_min_steps", 100),
                    batch_size=getattr(args, "hijack_batch_size", 16)
                )
                if round_result is not None:
                    all_hijack_results.append(round_result)

            # MIA 
            if getattr(args, "eval_mia", False):
                try:
                    from attacks.membership_inference import run_membership_inference_attack
                    from attacks.hijack import extract_proto_feature

                    logging.info(f'=== MIA attack (round {rnd + 1}) ===')
                    round_mia_results = []

                    for cid in select_list:
                        client_node = clients[cid]
                        if not hasattr(client_node, 'local_protos') or not client_node.local_protos:
                            logging.warning(f"[MIA] Client {cid}: no local prototypes; skipping.")
                            continue

                        mia_results = run_membership_inference_attack(
                            server_node=client_node,
                            glob_proto=client_node.local_protos,
                            extract_fn=extract_proto_feature,
                            args=args,
                            comprehensive=getattr(args, "mia_comprehensive", True),
                            target_fprs=getattr(args, "mia_target_fprs", [0.01, 0.001])
                        )

                        if mia_results is not None:
                            round_mia_results.append({'round': rnd + 1, 'client_id': cid, 'results': mia_results})

                    if round_mia_results:
                        all_mia_results.append({'round': rnd + 1, 'client_results': round_mia_results})
                        avg_auc = sum(r['results'].get('auc_macro', float('nan')) for r in round_mia_results) / len(round_mia_results)
                        avg_adv = sum(r['results'].get('advantage_macro', float('nan')) for r in round_mia_results) / len(round_mia_results)
                        logging.info(f"[MIA][Round {rnd + 1}] Avg AUC={avg_auc:.4f}, Avg Advantage={avg_adv:.4f}")

                except Exception as e:
                    logging.error(f"[MIA] Error in round {rnd + 1}: {e}")
                    import traceback
                    traceback.print_exc()

            # Server step: aggregate or fine-tune
            if args.method in ['mpft']:
                logging.info('=== Server fine-tune ===')
                server = Server_update(args, server, clients, select_list)
                logging.info(f'[Round {rnd + 1}] Server fine-tune done.')
            else:
                logging.info('=== Server prototype aggregation ===')
                server = Server_update(args, server, clients, select_list)
                logging.info(f'[Round {rnd + 1}] Server aggregation done.')

            # Client local training
            logging.info('=== Client update ===')
            clients, train_loss, train_acc = Client_update(args, rnd, clients, server, select_list)
            logging.info(f'Local update: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%')

        # Wall-clock time (kept as a single global measurement)
        federated_time = time.time() - t_global_start
        logging.info(f'Total elapsed training time: {federated_time:.2f}s')
        logging.info('==' * 20)

        # Validation
        accs = []
        for cid, node in clients.items():
            if args.method == 'fedpcl':
                logging.info('=== Client encode before validation ===')
                Client_encode(args, clients, select_list, flag='validation', round_idx=rnd + 1)
                acc = validate_fedpcl(args, node)
            else:
                acc = validate(args, node)

            accs.append(acc)
            logging.info(f'[Client Validation] C{cid} | test_acc={acc:.2f}%')

        avg_acc = sum(accs) / len(accs)
        all_avg_accs.append(avg_acc)
        logging.info(f"[Round {rnd + 1}] Average client test acc: {avg_acc:.4f}%")

    logging.info(f'Final average personalized test accuracy: {avg_acc:.2f}%')

    # Save hijack summary if enabled
    if all_hijack_results is not None and len(all_hijack_results) > 0:
        from attacks.hijack import save_all_rounds_summary
        save_all_rounds_summary(all_hijack_results, args.output_dir)

    # Save MIA summary if enabled
    if all_mia_results is not None and len(all_mia_results) > 0:
        logging.info("\n" + "=" * 70)
        logging.info(f"[MIA] All-round summary ({len(all_mia_results)} rounds)")
        logging.info("=" * 70)

        all_round_avg_auc = []
        all_round_avg_adv = []

        for rd in all_mia_results:
            round_num = rd['round']
            client_results = rd['client_results']
            if len(client_results) == 0:
                continue

            round_auc = sum(r['results'].get('auc_macro', float('nan')) for r in client_results) / len(client_results)
            round_adv = sum(r['results'].get('advantage_macro', float('nan')) for r in client_results) / len(client_results)
            all_round_avg_auc.append(round_auc)
            all_round_avg_adv.append(round_adv)
            logging.info(f"  Round {round_num:2d}: AUC={round_auc:.4f}, Advantage={round_adv:.4f}")

        if len(all_round_avg_auc) > 0:
            overall_avg_auc = sum(all_round_avg_auc) / len(all_round_avg_auc)
            overall_std_auc = np.std(all_round_avg_auc)
            overall_avg_adv = sum(all_round_avg_adv) / len(all_round_avg_adv)
            overall_std_adv = np.std(all_round_avg_adv)

            logging.info("=" * 70)
            logging.info(f"  Overall Avg AUC:       {overall_avg_auc:.4f} (±{overall_std_auc:.4f})")
            logging.info(f"  Overall Avg Advantage: {overall_avg_adv:.4f} (±{overall_std_adv:.4f})")
            logging.info("=" * 70)

            mia_results_path = os.path.join(args.output_dir, "mia_all_rounds_results.json")
            mia_output = {
                'overall_summary': {
                    'avg_auc': float(overall_avg_auc),
                    'std_auc': float(overall_std_auc),
                    'avg_advantage': float(overall_avg_adv),
                    'std_advantage': float(overall_std_adv),
                    'num_rounds': len(all_mia_results)
                },
                'per_round': all_mia_results
            }
            with open(mia_results_path, 'w') as f:
                json.dump(mia_output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            logging.info(f"[MIA] Results saved to: {mia_results_path}")

    logging.info("\n-- All rounds average client test accuracy --")
    for i, acc in enumerate(all_avg_accs, start=1):
        logging.info(f"Round {i:2d}: {acc:6.2f}%")

    metrics = {'args': vars(args), 'personalization_acc': avg_acc, 'times': {'total': federated_time}}
    metrics_path = os.path.join(args.output_dir, f'{args.method}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()